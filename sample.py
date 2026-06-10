import os
import sys
import argparse
import torch
import tiktoken

def get_args():
    parser = argparse.ArgumentParser(description="Sample/Inference script for Xeyronox 1")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Explain why the sky is blue.", help="Prompt to generate text for")
    parser.add_argument("--template-mode", type=str, default="final_only", 
                        choices=["final_only", "plan_answer", "draft_critique_revision", "principles_plan_final"],
                        help="Template format option matching the fine-tuned model training structure")
    parser.add_argument("--show-reasoning", action="store_true", 
                        help="If set, print full reasoning/principles/critiques alongside final answer. Default: final answer only.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling constraint")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P sampling constraint")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu, cuda, mps)")
    return parser.parse_args()

def format_prompt(prompt: str, mode: str) -> str:
    prompt = prompt.strip()
    if mode == "final_only":
        return f"### Prompt:\n{prompt}\n\n### Response:\n"
    elif mode == "plan_answer":
        return f"### Prompt:\n{prompt}\n\n### Plan:\n"
    elif mode == "draft_critique_revision":
        return f"### Prompt:\n{prompt}\n\n### Draft:\n"
    elif mode == "principles_plan_final":
        return f"### Prompt:\n{prompt}\n\n### Principles:\n"
    return f"### Prompt:\n{prompt}\n\n### Response:\n"

def parse_generation(full_text: str, mode: str, show_reasoning: bool) -> str:
    """
    Parses the full generated text and splits the final response from
    the structured critique/planning metadata based on template tags.
    """
    # If the user explicitly wants everything, return the raw generated text
    if show_reasoning:
        return full_text
        
    # Otherwise, extract the final answer (the text following the last '### Response:')
    target_tag = "### Response:\n"
    if target_tag in full_text:
        return full_text.split(target_tag)[-1].strip()
        
    return full_text.strip()

def run_mock_generation(prompt: str, mode: str, show_reasoning: bool):
    """
    Simulates model inference outputs for validation when model.py is not yet present.
    """
    print(f"\n[INFO] model.py not found. Running in Mock Inference Mode.")
    formatted = format_prompt(prompt, mode)
    print("=" * 60)
    print("Formatted Prompt Input:")
    print(formatted)
    print("=" * 60)
    
    # Generate mock outputs corresponding to SFT template schemas
    mock_responses = {
        "Explain why the sky is blue.": {
            "principles": "- Be helpful\n- Be concise\n- Be structured",
            "plan": "PLAN: Rayleigh scattering, gas molecules, wavelength scattering.\nANSWER: Rayleigh scattering causes blue light to disperse widely in the atmosphere.",
            "draft": "The sky is blue because light scattering...",
            "critique": "This answer is too brief and lacks structure. It should explain Rayleigh scattering clearly and calmly.",
            "response": "The sky appears blue due to Rayleigh scattering. As sunlight passes through the Earth's atmosphere, gases scatter shorter wavelengths of light (like blue and violet) more than longer wavelengths (like red and yellow). Because our eyes are more sensitive to blue light and violet light is absorbed in the upper atmosphere, we perceive the sky as blue."
        },
        "Is it safe to eat raw eggs?": {
            "principles": "- Be honest about uncertainty\n- Refuse harmful requests clearly\n- Avoid evasive refusals",
            "plan": "PLAN: Explain Salmonella risk, state susceptibility factors, advise pasteurization.\nANSWER: Salmonella is a risk; pasteurization reduces danger.",
            "draft": "No, you shouldn't eat raw eggs because of Salmonella, which can make you sick.",
            "critique": "The refusal is too aggressive. It should state facts clearly and present risk factors calmly.",
            "response": "Consuming raw eggs carries a low but present risk of Salmonella infection, which can cause food poisoning. Certain groups, such as pregnant women, the elderly, and individuals with weakened immune systems, should avoid raw eggs. Using pasteurized eggs significantly minimizes this risk."
        }
    }
    
    # Fallback default mock record if prompt is custom
    default_mock = {
        "principles": "- Be helpful\n- Be honest\n- Be concise",
        "plan": "PLAN: Formulate structured response.\nANSWER: Respond directly to custom prompt.",
        "draft": "Draft response placeholder.",
        "critique": "Critique placeholder.",
        "response": f"This is a mock response from the Xeyronox 1 assistant for: '{prompt}'."
    }
    
    data = mock_responses.get(prompt.strip(), default_mock)
    
    # Construct complete generated output
    if mode == "final_only":
        full_output = formatted + data["response"]
    elif mode == "plan_answer":
        full_output = formatted + data["plan"] + "\n\n### Response:\n" + data["response"]
    elif mode == "draft_critique_revision":
        full_output = formatted + data["draft"] + "\n\n### Critique:\n" + data["critique"] + "\n\n### Response:\n" + data["response"]
    elif mode == "principles_plan_final":
        full_output = formatted + data["principles"] + "\n\n### Plan:\n" + data["plan"] + "\n\n### Response:\n" + data["response"]
    else:
        full_output = formatted + data["response"]
        
    parsed_output = parse_generation(full_output, mode, show_reasoning)
    print("\nGenerated Output:")
    print(parsed_output)
    print("=" * 60)

def main():
    args = get_args()
    
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    
    model_exists = os.path.exists("model.py")
    if not model_exists or not os.path.exists(args.checkpoint):
        # Fall back to mock generation for testing SFT templates out-of-the-box
        run_mock_generation(args.prompt, args.template_mode, args.show_reasoning)
        return

    # Real Inference Setup
    from model import GPT
    from config.model_config import ModelConfig
    
    print(f"Loading checkpoint from: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Check if config is bundled in checkpoint, otherwise load default config
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        cfg = ModelConfig(n_layer=12, n_head=12, n_embd=768, block_size=256, vocab_size=50257) # fallback default
        
    model = GPT(cfg)
    
    # Strip DDP 'module.' or '_orig_mod.' prefixes if present
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    from collections import OrderedDict
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("_orig_mod."):
            name = name[10:]
        clean_state_dict[name] = v
        
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    # Format the input prompt matching the target template mode
    formatted_prompt = format_prompt(args.prompt, args.template_mode)
    
    # Initialize tokenizer
    try:
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = tiktoken.get_encoding("gpt2")
        
    input_ids = torch.tensor(enc.encode(formatted_prompt, allowed_special="all"), dtype=torch.long, device=device)[None, :]
    
    print(f"Generating (temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p})...")
    with torch.no_grad():
        # Autoregressive generation loop
        # model.generate method should support temperature, top_k, and optional top_p
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
    generated_text = enc.decode(output_ids[0].tolist())
    parsed_output = parse_generation(generated_text, args.template_mode, args.show_reasoning)
    
    print("\n" + "=" * 60)
    print("Output:")
    print(parsed_output)
    print("=" * 60)

if __name__ == "__main__":
    main()
