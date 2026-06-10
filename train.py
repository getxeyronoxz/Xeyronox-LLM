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
from data_utils import StreamingTokenDataset

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain Xeyronox 1 small language model")
    parser.add_argument("--config", type=str, default="config/config_cpu_debug.json", help="Path to config JSON")
    parser.add_argument("--data-path", type=str, default="data/sample_text.txt", help="Path or glob pattern for text files")
    parser.add_argument("--is-hf", action="store_true", help="Whether data-path is a Hugging Face dataset name")
    parser.add_argument("--hf-split", type=str, default="train", help="Split name to use if streaming HF dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch size if specified")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--dry-run", action="store_true", help="If True, only test the dataset pipeline throughput and exit")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    return parser.parse_args()

class DDPDatasetWrapper(torch.utils.data.IterableDataset):
    """
    Wraps an IterableDataset to shard data stream across multiple DDP ranks.
    Ensures different processes train on non-overlapping token blocks.
    """
    def __init__(self, dataset, rank, world_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        block_idx = 0
        for x, y in self.dataset:
            if block_idx % self.world_size == self.rank:
                yield x, y
            block_idx += 1

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
            
        master_process = ddp_rank == 0 # main rank handles logging, checkpoints, etc.
        seed_offset = ddp_rank
    else:
        # Vanilla single-GPU or CPU execution
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
        print("Xeyronox 1 - Training Initialization")
        print("=" * 60)
        print(f"DDP Mode: {'Enabled' if ddp else 'Disabled'}")
        if ddp:
            print(f"World Size: {ddp_world_size} | Local Rank: {ddp_local_rank}")
        print(f"Primary Device: {device}")

    # Set seed based on offset to ensure different shuffling patterns per rank
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
    
    # Training hyperparams (defaults)
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    
    if master_process:
        print(f"Initializing streaming dataset pipeline...")
        print(f"Source: {args.data_path}")
        print(f"Block size: {cfg.block_size}")
        print(f"Batch size: {batch_size}")
    
    # Check if files matching data-path exist (unless it's HF)
    if not args.is_hf and master_process:
        import glob
        matched = glob.glob(args.data_path)
        if not matched:
            print(f"\nNo local files matched '{args.data_path}'. Creating a temporary sample corpus for verification...")
            os.makedirs(os.path.dirname(os.path.abspath(args.data_path)), exist_ok=True)
            with open(args.data_path, 'w', encoding='utf-8') as f:
                f.write("The quick brown fox jumps over the lazy dog. " * 1000)
                f.write("\nHello world! This is a test file for training a small nanoGPT model. " * 500)
            print(f"Created sample text file at {args.data_path}")

    # Ensure all processes wait for file creation in DDP mode
    if ddp:
        torch.distributed.barrier()

    # Initialize datasets
    train_dataset = StreamingTokenDataset(
        source=args.data_path,
        tokenizer_name="gpt2",
        block_size=cfg.block_size,
        split="train",
        is_hf=args.is_hf
    )
    
    val_dataset = StreamingTokenDataset(
        source=args.data_path,
        tokenizer_name="gpt2",
        block_size=cfg.block_size,
        split="val",
        is_hf=args.is_hf
    )
    
    # Shard streams across DDP ranks if world_size > 1
    if ddp_world_size > 1:
        train_dataset = DDPDatasetWrapper(train_dataset, ddp_rank, ddp_world_size)
        val_dataset = DDPDatasetWrapper(val_dataset, ddp_rank, ddp_world_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers)
    
    # Check if model.py exists to import the actual model
    model_exists = os.path.exists("model.py")
    
    if args.dry_run or not model_exists:
        if master_process:
            if not model_exists:
                print("\n[INFO] model.py not found. Running in Dry Run / Pipeline Check mode.")
            else:
                print("\n[INFO] Running in Dry Run / Pipeline Check mode as requested.")
            print("Iterating through train loader batches to test dataset pipeline throughput...")
        
        start_time = time.time()
        max_batches = 50
        batch_count = 0
        total_tokens = 0
        
        try:
            for x, y in train_loader:
                batch_count += 1
                x = x.to(device)
                y = y.to(device)
                total_tokens += x.numel()
                
                if master_process and (batch_count % 10 == 0 or batch_count == 1):
                    print(f"  Train Batch {batch_count}: inputs shape {x.shape}, targets shape {y.shape}")
                
                if batch_count >= max_batches:
                    break
            
            # Check validation loader
            if master_process:
                print("Checking validation loader...")
            val_iter = iter(val_loader)
            try:
                x_val, y_val = next(val_iter)
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                if master_process:
                    print(f"  Val Batch 1: inputs shape {x_val.shape}, targets shape {y_val.shape}")
            except StopIteration:
                if master_process:
                    print("  [WARNING] Validation loader is empty.")
                
        except KeyboardInterrupt:
            if master_process:
                print("Interrupted by user.")
        except Exception as e:
            print(f"[Rank {ddp_rank} ERROR] Pipeline execution failed: {e}")
            if ddp:
                destroy_process_group()
            sys.exit(1)
            
        duration = time.time() - start_time
        
        # Accumulate metrics across DDP ranks to report aggregate stats on master process
        if ddp:
            tokens_tensor = torch.tensor([total_tokens], dtype=torch.long, device=device)
            torch.distributed.reduce(tokens_tensor, dst=0)
            total_tokens = tokens_tensor.item()
            
        if master_process:
            print("\n" + "=" * 60)
            print("Pipeline Check Completed Successfully")
            print("=" * 60)
            print(f"Total Batches Processed per rank: {batch_count}")
            print(f"Total Tokens Loaded (All ranks):  {total_tokens:,}")
            print(f"Total Duration:                   {duration:.4f} seconds")
            if duration > 0:
                print(f"Throughput:                       {total_tokens / duration:,.2f} tokens/sec")
            print("=" * 60)
        
        if ddp:
            destroy_process_group()
        return

    # 3. Real DDP/Single-Device Training Loop
    if master_process:
        print("\n[INFO] model.py found! Starting actual training...")
        
    from model import GPT
    model = GPT(cfg)
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
    
    start_time = time.time()
    for step in range(1, cfg.max_iters + 1):
        # Periodic evaluation and checkpointing
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
                print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
            # Save the best model
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
                    torch.save(checkpoint, 'checkpoint.pt')
                    print(f"  Saved best checkpoint to 'checkpoint.pt' (val_loss: {best_val_loss:.4f})")
            
        # Training iteration step
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
            print(f"Step {step}/{cfg.max_iters} | loss: {loss_accum:.4f} | time: {iter_time:.2f}s")
            start_time = time.time()
            
    if master_process:
        print("\nPretraining completed.")
        
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()
