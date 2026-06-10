# Xeyronox 1 - NanoGPT-Style Small Language Model

A small, decoder-only text language model built from scratch, targeting between 10M and 40M parameters (with a primary target of 20M to 30M). The model is designed to train on limited hardware (including low-memory local CPUs and Kaggle dual T4 GPUs) and is instruction-tuned and aligned using synthetic constitutional data.

This project is a pure text language model. It does not implement Mixture of Experts (MoE), retrieval-augmented generation (RAG), tool use, live self-reflection loops, or multimodal features.

---

## Project Structure

```
├── config/
│   ├── model_config.py      # Dataclass defining the model configuration and parameter estimation
│   ├── config_cpu_debug.json # Tiny config for low-memory CPU dry runs (~2.9M params)
│   ├── config_10m.json      # Config for debug runs (~10.6M params)
│   ├── config_25m.json      # Config for main training run (~27.7M params)
│   ├── config_40m.json      # Config for upper-bound model (~36.9M params)
│   └── verify_configs.py    # Python script to validate config structure and parameter bounds
├── AGENTS.md                # Project requirements, roadmap, and development instructions
├── README.md                # Project overview, directory layout, and setup instructions
```

---

## Getting Started

### 1. Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA support if training on GPUs)
- standard dependencies (will be detailed in `requirements.txt` in subsequent stages)

### 2. Verify Configs
Verify the initial model configurations and parameters:
```bash
python config/verify_configs.py
```

---

## Planned Development Stages

### Stage 1: Base Pretraining (`train.py`)
- Standard autoregressive next-token prediction training on clean text corpora.
- Implement incremental tokenizer and data stream processor (`data_utils.py`) supporting Hugging Face iterable datasets without loading full corpora into RAM.

### Stage 2: Supervised Instruction Tuning (`finetune.py`)
- Instruction tuning on prompt-response pairs.
- Target a calm, structured, helpful, and concise response tone.

### Stage 3: Synthetic Constitutional Fine-Tuning (`finetune.py`)
- Self-critique and revision loop utilizing a small explicit constitution.
- The student model is trained on the final revised outputs of the teacher draft critique-and-revision cycle.

### Stage 4: Evaluation and Structured Formatting (`evaluate.py`, `sample.py`)
- Lightweight evaluation of training loss, approximate perplexity, instruction following, and constitutional compliance.
- Supports structured generation templates (e.g., `FINAL only`, `PLAN -> ANSWER`, `DRAFT -> CRITIQUE -> REVISION`).

---

## How to Run

### 1. Base Pretraining (DDP Optional)
To launch pretraining:
- **CPU (local debug)**:
  ```bash
  python train.py --config config/config_cpu_debug.json --data-path data/sample_text.txt
  ```
- **Kaggle 2xT4 (torchrun)**:
  ```bash
  torchrun --standalone --nproc_per_node=2 train.py --config config/config_25m.json --data-path data/sample_text.txt
  ```

### 2. Synthetic Constitutional SFT
To launch SFT:
```bash
python finetune.py --template-mode final_only --dry-run
```
For alternative template structures (e.g. including short reasoning plans):
```bash
python finetune.py --template-mode plan_answer --dry-run
```

### 3. Inference and Sampling
To sample responses from the model:
- **Default (Final Response Only)**:
  ```bash
  python sample.py --template-mode plan_answer --prompt "Explain why the sky is blue."
  ```
- **Reasoning-Visible Mode**:
  ```bash
  python sample.py --template-mode plan_answer --show-reasoning --prompt "Explain why the sky is blue."
  ```

### 4. Metrics & Compliance Evaluation
To run the evaluation script and qualitative regression tests:
```bash
python evaluate.py --template-mode final_only
```
For structured modes:
```bash
python evaluate.py --template-mode plan_answer
```
