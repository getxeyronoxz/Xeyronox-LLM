import os
import sys
from model_config import ModelConfig

def main():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    configs = [
        ("CPU Debug Model", "config_cpu_debug.json", 0, 5_000_000),
        ("10M Debug Model", "config_10m.json", 8_000_000, 15_000_000),
        ("25M Main Model", "config_25m.json", 20_000_000, 30_000_000),
        ("40M Upper-bound Model", "config_40m.json", 35_000_000, 45_000_000),
    ]

    print("=" * 60)
    print("Xeyronox 1 - Configurations Verification Tool")
    print("=" * 60)

    has_errors = False

    for name, filename, min_params, max_params in configs:
        path = os.path.join(config_dir, filename)
        print(f"\nChecking: {name} ({filename})")
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}")
            has_errors = True
            continue

        try:
            cfg = ModelConfig.load_json(path)
            
            # Constraints checking
            # n_embd must be divisible by n_head
            if cfg.n_embd % cfg.n_head != 0:
                print(f"  [ERROR] n_embd ({cfg.n_embd}) is not divisible by n_head ({cfg.n_head})!")
                has_errors = True
            
            # Block size limits
            if cfg.block_size < 16 or cfg.block_size > 1024:
                print(f"  [WARNING] Unusual block_size: {cfg.block_size}")

            params_tied = cfg.estimate_parameters(tie_weights=True)
            params_untied = cfg.estimate_parameters(tie_weights=False)

            print(f"  Vocab Size:    {cfg.vocab_size}")
            print(f"  Context Length:{cfg.block_size}")
            print(f"  Layers (n_lay):{cfg.n_layer}")
            print(f"  Heads (n_head):{cfg.n_head}")
            print(f"  Embed dim (D): {cfg.n_embd}")
            print(f"  Tied Params:   {params_tied:,}")
            print(f"  Untied Params: {params_untied:,}")

            if min_params <= params_tied <= max_params:
                print(f"  [PASS] Parameter count ({params_tied:,}) is within target range ({min_params:,} - {max_params:,})")
            else:
                print(f"  [WARNING] Parameter count ({params_tied:,}) is outside target range ({min_params:,} - {max_params:,})")

        except Exception as e:
            print(f"  [ERROR] Failed to load/verify config: {e}")
            has_errors = True

    print("\n" + "=" * 60)
    if has_errors:
        print("Verification completed with ERRORS.")
        sys.exit(1)
    else:
        print("All configurations verified successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
