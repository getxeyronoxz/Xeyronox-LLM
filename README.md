# Xeyronox 1 - NanoGPT-Style Small Language Model

A small, decoder-only text language model built from scratch, targeting between 10M and 40M parameters (with a primary target of 20M to 30M). The model is designed to train on limited hardware (including low-memory local CPUs and Kaggle dual T4 GPUs) and is instruction-tuned and aligned using synthetic constitutional data.

This project is a pure text language model. It does not implement Mixture of Experts (MoE), retrieval-augmented generation (RAG), tool use, live self-reflection loops, or multimodal features.

---

## Project Structure

```
├── config/
│   ├── model_config.py        # Dataclass defining the model configuration and parameter estimation
│   ├── config_cpu_debug.json  # Tiny config for low-memory CPU dry runs (~2.9M params)
│   ├── config_10m.json        # Config for debug runs (~10.6M params / ~25M with GPT-2 vocab)
│   ├── config_25m.json        # Config for main training run (~27.7M params / ~40.59M with GPT-2 vocab)
│   ├── config_40m.json        # Config for upper-bound model (~36.9M params / ~54.09M with GPT-2 vocab)
│   └── verify_configs.py      # Python script to validate config structure and parameter bounds
├── data_utils.py              # Out-of-core streaming data loaders with automatic 90/10 train/val partitioning
├── ddp_launcher.py            # Windows DDP rendezvous and launcher helper bypassing libuv issues
├── evaluate.py                # Quantitative validation loss/perplexity and qualitative regression suite
├── finetune.py                # Supervised Fine-Tuning (SFT) script using Constitutional AI templates
├── model.py                   # Decoder-only GPT-style autoregressive transformer model
├── sample.py                  # Inference script supporting structured outputs and reasoning mode
├── train.py                   # Base pretraining training loop supporting DDP multi-GPU scaling
├── AGENTS.md                  # Project requirements, roadmap, and development instructions
├── README.md                  # Project overview, directory layout, and setup instructions
├── requirements.txt           # Pip dependencies list
```

---

## Features & Implementation Details

* **On-the-Fly Tokenization & Streaming**: The dataset pipeline streams text incrementally, tokenizes via BPE (`tiktoken` GPT-2 tokenizer), and discards raw text immediately to maintain a sub-2GB RAM footprint.
* **Leakage-Free Partitioning**: Single-split Hugging Face datasets (like Dolly-15k) are automatically sharded on-the-fly into 90% training and 10% validation subsets.
* **Distributed Training (DDP)**: Supports Multi-GPU DDP (`torchrun`) for scaling. Includes a custom launcher patch (`ddp_launcher.py`) that bypasses Windows compilation limitations for local debugging.
* **Constitutional Alignment**: Supports four SFT template modes for alignment (including visible chain-of-thought plans, critiques, and final-only formats).

---

## How to Run

Before running, make sure to install all dependencies:
```bash
pip install -r requirements.txt
```

Verify your model configurations:
```bash
python config/verify_configs.py
```

### 1. Base Pretraining (`train.py`)
Pretrain the model on a raw text corpus (next-token prediction):
* **Local CPU (Debug Mode)**:
  ```bash
  python train.py --config config/config_cpu_debug.json --data-path data/sample_text.txt
  ```
* **Kaggle 2xT4 GPUs (Distributed DDP)**:
  ```bash
  torchrun --nproc_per_node=2 train.py --config config/config_40m.json --data-path roneneldan/TinyStories --is-hf --batch-size 8
  ```

### 2. Supervised Fine-Tuning (`finetune.py`)
Align the model using instruction datasets:
* **Local CPU (Dry-Run)**:
  ```bash
  python finetune.py --config config/config_cpu_debug.json --dry-run
  ```
* **Kaggle 2xT4 GPUs (SFT on Code Instructions)**:
  ```bash
  # Sets Hugging Face token for authentication
  export HF_TOKEN="your_hf_token_here"
  
  torchrun --nproc_per_node=2 finetune.py --config config/config_40m.json --data-path sahil2801/CodeAlpaca-20k --template-mode final_only --batch-size 8
  ```

### 3. Metrics & Adherence Evaluation (`evaluate.py`)
Evaluate your model quantitatively (Validation Loss and Perplexity) and qualitatively (Adherence, Refusal Safety, and Length constraints):
```bash
python evaluate.py --checkpoint sft_checkpoint.pt --template-mode final_only
```

### 4. Sampling / Inference (`sample.py`)
Generate text with the aligned model:
* **Default Mode (Final Answer Only)**:
  ```bash
  python sample.py --checkpoint sft_checkpoint.pt --template-mode final_only --prompt "Write a Python function to check if a number is prime."
  ```
* **Reasoning Mode (Visible Plan & Thinking)**:
  ```bash
  python sample.py --checkpoint sft_checkpoint.pt --template-mode plan_answer --show-reasoning --prompt "Write a Python function to check if a number is prime."
  ```

---

## Parameters Target Mappings
Due to BPE vocabulary size expanding from `16,384` to `50,257` dynamically at runtime, the parameter mappings are:
* `config_25m.json` -> **40.59 Million parameters** (Target main model size).
* `config_40m.json` -> **54.09 Million parameters** (Upper-bound model size).

