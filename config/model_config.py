import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ModelConfig:
    # Model Hyperparameters
    vocab_size: int = 16384
    block_size: int = 256  # context length
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit faster and cleaner

    # Training Parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    max_iters: int = 100
    eval_interval: int = 20
    eval_iters: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelConfig':
        # Filter keys to match fields
        fields = cls.__dataclass_fields__
        filtered_data = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered_data)

    @classmethod
    def load_json(cls, path: str) -> 'ModelConfig':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    def estimate_parameters(self, tie_weights: bool = True) -> int:
        """
        Estimates the total number of parameters in the model.
        Formula for decoder-only transformer:
        - Token Embedding (wte): V * D
        - Position Embedding (wpe): S * D (assumes absolute, nanoGPT-style)
        - For each layer:
            - Attn (Q, K, V): 3 * D * D (+ 3 * D if bias)
            - Attn output proj: D * D (+ D if bias)
            - MLP first linear: D * (4 * D) (+ 4 * D if bias)
            - MLP second linear: (4 * D) * D (+ D if bias)
            - LayerNorm 1: 2 * D
            - LayerNorm 2: 2 * D
        - Final LayerNorm: 2 * D
        - LM Head (if untied): V * D
        """
        v = self.vocab_size
        s = self.block_size
        d = self.n_embd
        l = self.n_layer
        bias_multiplier = 1 if self.bias else 0

        wte_params = v * d
        wpe_params = s * d
        
        # Attention parameters
        # QKV projection + output projection
        attn_weights = 4 * d * d
        attn_biases = 4 * d * bias_multiplier if self.bias else 0
        
        # MLP parameters (4x expansion)
        mlp_weights = 8 * d * d
        mlp_biases = 5 * d * bias_multiplier if self.bias else 0
        
        # LayerNorms: 2 per layer (each has weight + bias, so 2 * 2 * d)
        ln_params = 4 * d
        
        per_layer_params = attn_weights + attn_biases + mlp_weights + mlp_biases + ln_params
        
        total_layers = l * per_layer_params
        
        final_ln = 2 * d
        
        lm_head = 0 if tie_weights else (v * d)
        
        total = wte_params + wpe_params + total_layers + final_ln + lm_head
        return total
