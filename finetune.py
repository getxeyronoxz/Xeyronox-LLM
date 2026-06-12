import os
import sys

# Patch TCPStore to disable libuv on Windows before importing/running torch distributed
if os.name == 'nt':
    try:
        import ctypes
        try:
            ctypes.windll.kernel32.SetEnvironmentVariableW('USE_LIBUV', '0')
        except Exception:
            pass
            
        try:
            ctypes.CDLL('msvcrt')._wputenv('USE_LIBUV=0')
        except Exception:
            pass
            
        import torch
        import torch.distributed
        import torch.distributed.rendezvous
        import torch.distributed.elastic.rendezvous.c10d_rendezvous_backend
        
        orig_tcp_store = torch.distributed.TCPStore
        
        def patched_tcp_store(*args, **kwargs):
            # Enforce use_libuv=False to bypass Windows libuv compilation issues
            kwargs['use_libuv'] = False
            return orig_tcp_store(*args, **kwargs)
            
        torch.distributed.TCPStore = patched_tcp_store
        
        # Dynamically patch sys.modules to propagate the patched TCPStore
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, '__dict__') and 'TCPStore' in mod.__dict__:
                if mod.__dict__['TCPStore'] is orig_tcp_store:
                    mod.__dict__['TCPStore'] = patched_tcp_store
    except Exception as e:
        sys.stderr.write(f"PATCH ERROR in child process: {e}\n")
        sys.stderr.flush()

os.environ["USE_LIBUV"] = "0"
import sys
import time
import argparse
import torch
from torch.utils.data import DataLoader
from config.model_config import ModelConfig
from data_utils import ConstitutionalDataset

def get_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Xeyronox 1 using Constitutional AI templates")
    parser.add_argument("--config", type=str, default="config/config_cpu_debug.json", help="Path to config JSON")
    parser.add_argument("--data-path", type=str, default="data/constitutional_data.jsonl", help="Path to JSONL data file")
    parser.add_argument("--template-mode", type=str, default="final_only", 
                        choices=["final_only", "plan_answer", "draft_critique_revision", "principles_plan_final"],
                        help="Template mode for formatting SFT dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size if specified")
    parser.add_argument("--max-iters", type=int, default=None, help="Override config max_iters if specified")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--dry-run", action="store_true", help="If True, only test the SFT dataset pipeline and exit")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    return parser.parse_args()

def create_sample_jsonl(filepath: str):
    import json
    samples = [
        {
            "prompt": "Explain why the sky is blue.",
            "draft": "The sky is blue because light scattering...",
            "critique": "This answer is too brief and lacks structure. It should explain Rayleigh scattering clearly and calmly.",
            "revision": "The sky appears blue due to a phenomenon called Rayleigh scattering. As sunlight passes through the Earth's atmosphere, gases scatter shorter wavelengths of light (like blue and violet) more than longer wavelengths (like red and yellow). Because our eyes are more sensitive to blue light and violet light is absorbed in the upper atmosphere, we perceive the sky as blue.",
            "final_answer": "The sky is blue due to Rayleigh scattering, where air molecules scatter shorter wavelengths of sunlight (blue/violet) more effectively than longer wavelengths (red/yellow).",
            "short_reasoning": "PLAN: Rayleigh scattering, gas molecules, wavelength scattering.\nANSWER: Rayleigh scattering causes blue light to disperse widely in the atmosphere.",
            "principles_used": ["Be helpful", "Be concise", "Be structured"]
        },
        {
            "prompt": "Is it safe to eat raw eggs?",
            "draft": "No, you shouldn't eat raw eggs because of Salmonella, which can make you sick.",
            "critique": "The refusal is too aggressive. It should state facts clearly and present the uncertainty and risk factors calmly without lecturing.",
            "revision": "Consuming raw eggs carries a risk of Salmonella infection, which can cause food poisoning. While the risk of an egg containing Salmonella is relatively low, certain groups such as pregnant women, elderly individuals, and those with weakened immune systems should avoid raw eggs. Using pasteurized eggs significantly reduces this risk.",
            "final_answer": "Consuming raw eggs carries a low but present risk of Salmonella infection. Pasteurized eggs are recommended for vulnerable populations to minimize this risk.",
            "short_reasoning": "PLAN: Explain Salmonella risk, state susceptibility factors, advise pasteurization.\nANSWER: Salmonella is a risk; pasteurization reduces danger.",
            "principles_used": ["Be honest about uncertainty", "Refuse harmful requests clearly", "Avoid evasive refusals"]
        },
        {
            "prompt": "Tell me about quantum computing.",
            "draft": "Quantum computing is super fast and uses qubits instead of bits to do things normal computers can't.",
            "critique": "The response is overly enthusiastic and lacks technical precision. It should explain superposition and entanglement calmly and concisely.",
            "revision": "Quantum computing utilizes the principles of quantum mechanics to process information. Unlike classical computers that represent data as bits (0s or 1s), quantum computers use qubits. Qubits can exist in a superposition of states and can be entangled, allowing them to solve certain complex computational problems much faster than classical systems.",
            "final_answer": "Quantum computing leverages superposition and entanglement via qubits to perform parallel calculations, offering speedups for specific complex problems.",
            "short_reasoning": "PLAN: Define qubits, explain superposition and entanglement, mention computational limits.\nANSWER: Uses quantum mechanics principles for specific computations.",
            "principles_used": ["Be helpful", "Avoid unnecessary verbosity", "Prefer concise, well-structured answers"]
        }
    ]
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

def main():
    args = get_args()
    
    # 1. Distributed Data Parallel (DDP) Setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # checks if launched via torchrun
    if ddp:
        from torch.distributed import init_process_group, destroy_process_group
        backend = 'gloo' if (os.name == 'nt' or not torch.cuda.is_available()) else 'nccl'
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        
        if torch.cuda.is_available():
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
            
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        
        if args.device is not None:
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    if master_process:
        print("=" * 60)
        print("Xeyronox 1 - Supervised Fine-Tuning (SFT)")
        print("=" * 60)
        print(f"DDP Mode:      {'Enabled' if ddp else 'Disabled'}")
        if ddp:
            print(f"World Size:    {ddp_world_size} | Local Rank: {ddp_local_rank}")
        print(f"Template Mode: {args.template_mode}")
        print(f"Primary Device: {device}")

    torch.manual_seed(1337 + seed_offset)
    
    # Load model configuration
    if not os.path.exists(args.config):
        if master_process:
            print(f"Error: Config path {args.config} not found.")
        if ddp:
            destroy_process_group()
        sys.exit(1)
        
    cfg = ModelConfig.load_json(args.config)
    cfg.vocab_size = 50257 # Match GPT-2 tokenizer vocab size
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    # Check if files matching data-path exist
    is_hf = not args.data_path.endswith(".jsonl") and not os.path.exists(args.data_path) and "/" in args.data_path
    if not is_hf and master_process:
        if not os.path.exists(args.data_path):
            print(f"\nLocal SFT data file not found. Generating sample records at: {args.data_path}")
            create_sample_jsonl(args.data_path)

    # Ensure all processes wait for file creation in DDP mode
    if ddp:
        torch.distributed.barrier()

    if master_process:
        print(f"Initializing Constitutional Dataset...")
        print(f"Source:        {args.data_path}")
        print(f"Block size:    {cfg.block_size}")
        print(f"Batch size:    {batch_size}")

    # Initialize Constitutional datasets
    train_dataset = ConstitutionalDataset(
        source=args.data_path,
        tokenizer_name="gpt2",
        block_size=cfg.block_size,
        template_mode=args.template_mode,
        split="train"
    )
    
    val_dataset = ConstitutionalDataset(
        source=args.data_path,
        tokenizer_name="gpt2",
        block_size=cfg.block_size,
        template_mode=args.template_mode,
        split="val"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers)

    model_exists = os.path.exists("model.py")
    
    if args.dry_run or not model_exists:
        if master_process:
            if not model_exists:
                print("\n[INFO] model.py not found. Running in Dry Run / SFT Pipeline Check mode.")
            else:
                print("\n[INFO] Running in Dry Run / SFT Pipeline Check mode as requested.")
            print("Iterating SFT DataLoader to verify formatting and tokenization...")
            
        start_time = time.time()
        max_batches = 10
        batch_count = 0
        total_tokens = 0
        
        try:
            for x, y in train_loader:
                batch_count += 1
                x = x.to(device)
                y = y.to(device)
                total_tokens += x.numel()
                
                if master_process and (batch_count % 2 == 0 or batch_count == 1):
                    print(f"  SFT Batch {batch_count}: inputs shape {x.shape}, targets shape {y.shape}")
                    # Decode sample from tokenizer to verify template mode visually
                    from tiktoken import get_encoding
                    enc = get_encoding("gpt2")
                    decoded_input = enc.decode(x[0].tolist())
                    print("-" * 50)
                    print(f"  Sample Decoded Input (Batch {batch_count}):")
                    print(decoded_input)
                    print("-" * 50)
                
                if batch_count >= max_batches:
                    break
                    
            if master_process:
                print("Checking validation loader...")
            val_iter = iter(val_loader)
            try:
                x_val, y_val = next(val_iter)
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                if master_process:
                    print(f"  SFT Val Batch 1: inputs shape {x_val.shape}, targets shape {y_val.shape}")
            except StopIteration:
                if master_process:
                    print("  [INFO] Validation loader is empty (expected for tiny datasets).")
                    
        except KeyboardInterrupt:
            if master_process:
                print("Interrupted by user.")
        except Exception as e:
            print(f"[Rank {ddp_rank} ERROR] SFT Pipeline check failed: {e}")
            if ddp:
                destroy_process_group()
            sys.exit(1)
            
        duration = time.time() - start_time
        
        if ddp:
            tokens_tensor = torch.tensor([total_tokens], dtype=torch.long, device=device)
            torch.distributed.reduce(tokens_tensor, dst=0)
            total_tokens = tokens_tensor.item()
            
        if master_process:
            print("\n" + "=" * 60)
            print("SFT Pipeline Check Completed Successfully")
            print("=" * 60)
            print(f"Template Mode Selected:          {args.template_mode}")
            print(f"Total SFT Batches per rank:      {batch_count}")
            print(f"Total SFT Tokens (All ranks):    {total_tokens:,}")
            print(f"Duration:                        {duration:.4f} seconds")
            print("=" * 60)
            
        if ddp:
            destroy_process_group()
        return

    # Real SFT tuning loop
    if master_process:
        print("\n[INFO] model.py found! Starting actual SFT...")
        
    from model import GPT
    model = GPT(cfg)
    
    # Load pretrained checkpoint if available
    checkpoint_path = "checkpoint.pt"
    if os.path.exists(checkpoint_path):
        if master_process:
            print(f"Loading pretrained checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        
        from collections import OrderedDict
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            name = name[10:] if name.startswith("_orig_mod.") else name
            clean_state_dict[name] = v
        model.load_state_dict(clean_state_dict)
    else:
        if master_process:
            print("[WARNING] Pretrained checkpoint 'checkpoint.pt' not found. Starting SFT from scratch.")
            
    model = model.to(device)
    
    raw_model = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if 'cuda' in device:
            model = DDP(model, device_ids=[ddp_local_rank])
        else:
            model = DDP(model)
        raw_model = model.module
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    best_val_loss = 1e9
    train_iter = iter(train_loader)
    
    max_iters = args.max_iters if args.max_iters is not None else cfg.max_iters
    start_time = time.time()
    for step in range(1, max_iters + 1):
        # Periodic evaluation and SFT checkpoint saving
        if step % cfg.eval_interval == 0 or step == 1:
            losses = {}
            model.eval()
            for split_name, loader in [('train', train_loader), ('val', val_loader)]:
                loader_iter = iter(loader)
                eval_loss = 0.0
                eval_steps = 0
                for _ in range(cfg.eval_iters):
                    try:
                        bx, by = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(loader)
                        try:
                            bx, by = next(loader_iter)
                        except StopIteration:
                            break
                    bx = bx.to(device)
                    by = by.to(device)
                    with torch.no_grad():
                        logits, loss = model(bx, by)
                    eval_loss += loss.item()
                    eval_steps += 1
                losses[split_name] = eval_loss / max(1, eval_steps)
            
            if master_process:
                print(f"SFT Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
            # Save the best SFT model
            if losses.get('val', 1e9) < best_val_loss:
                best_val_loss = losses['val']
                if master_process:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': cfg,
                        'iter': step,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, 'sft_checkpoint.pt')
                    print(f"  Saved best SFT checkpoint to 'sft_checkpoint.pt' (val_loss: {best_val_loss:.4f})")
            
        # SFT training iteration step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        grad_accum_steps = cfg.gradient_accumulation_steps
        
        for micro_step in range(grad_accum_steps):
            try:
                bx, by = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                try:
                    bx, by = next(train_iter)
                except StopIteration:
                    break
            bx = bx.to(device)
            by = by.to(device)
            
            if ddp and micro_step < grad_accum_steps - 1:
                with model.no_sync():
                    logits, loss = model(bx, by)
                    loss = loss / grad_accum_steps
                    loss.backward()
            else:
                logits, loss = model(bx, by)
                loss = loss / grad_accum_steps
                loss.backward()
            loss_accum += loss.item()
            
        if cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
        optimizer.step()
        
        if master_process and (step % 10 == 0 or step == 1):
            iter_time = time.time() - start_time
            print(f"SFT Step {step}/{max_iters} | loss: {loss_accum:.4f} | time: {iter_time:.2f}s")
            start_time = time.time()
            
    if master_process:
        print("\nSFT Fine-Tuning completed.")
        
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
