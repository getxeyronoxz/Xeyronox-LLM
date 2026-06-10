# AGENTS.md

## Project identity
Build a small decoder-only text language model from scratch, targeting 10M to 40M parameters.

This project is **not** a frontier model system. It must remain a pure text model:
- No MoE.
- No tool use.
- No browser or retrieval.
- No agent loops.
- No multimodal features.
- No production serving stack.

The desired behavior is a calm, structured, helpful assistant with a response style loosely inspired by Claude-like text behavior, but implemented as a compact GPT-style model.

## Primary goal
Create a small language model that:
- Trains from scratch on limited hardware.
- Can be instruction-tuned.
- Can be shaped using synthetic constitutional data.
- Supports structured reasoning-style outputs.
- Runs across Kaggle GPUs and low-end local hardware.

Recommended first target:
- 20M to 30M parameters.

Allowed range:
- 10M to 40M parameters.

## Design principles
- Prefer simple, explicit code.
- Stay close to a nanoGPT-style codebase.
- Make minimal, readable modifications.
- Optimize for experimentation, not scale.
- Keep hardware behavior predictable.
- Default inference must show final answers only.
- Any visible reasoning mode must be optional.

## Architecture
Use a decoder-only transformer with these expectations:
- GPT-style autoregressive language model.
- BPE or similar subword tokenizer.
- Initial context length: 256.
- Optional later context length: 512.
- Model configs should cover:
  - ~10M debug model,
  - ~20M to 30M main model,
  - ~35M to 40M upper-bound model,
  - low-memory CPU debug model.

## Non-goals
Do not implement:
- mixture-of-experts,
- retrieval-augmented generation,
- live self-reflection loops at inference time,
- RLHF infrastructure,
- RLAIF training in the first version,
- hidden production orchestration systems,
- unnecessary framework complexity.

## Training stages
Implement training in stages.

### Stage 1: Base pretraining
- Train on clean text corpora.
- Focus on stable next-token prediction.
- Use quality over quantity.

### Stage 2: Supervised instruction tuning
- Train on prompt/response pairs.
- Improve helpfulness, structure, and clarity.
- Keep outputs concise and calm.

### Stage 3: Synthetic constitutional fine-tuning
Use a scaled-down Constitutional AI inspired workflow:
- generate a teacher draft,
- critique it against a short constitution,
- produce a revision,
- train the student on the revised result.

The original Constitutional AI method includes both a supervised phase and a reinforcement-learning phase; this project should only implement the supervised synthetic critique-and-revision pattern in the first version.

### Stage 4: Optional compact reasoning formatting
Add optional structured targets such as:
- PLAN -> ANSWER
- DRAFT -> CRITIQUE -> REVISED ANSWER
- PRINCIPLES USED -> SHORT REASONING -> FINAL

Do not force long verbose chain-of-thought on every sample.
Prefer short structured reasoning traces.

## Constitution
Use a small explicit constitution for teacher-generated data.
Suggested principles:
- Be helpful.
- Be honest about uncertainty.
- Do not fabricate facts.
- Refuse harmful or disallowed requests clearly.
- Avoid evasive refusals.
- Prefer concise, well-structured answers.
- Explain objections briefly when refusing.
- Avoid unnecessary verbosity.
- Keep visible reasoning compact when enabled.

## Output behavior
The model should learn these default answer traits:
- calm tone,
- structured response,
- concise wording,
- uncertainty when unsure,
- direct refusal when needed,
- no unnecessary self-reference.

Default inference output:
- final answer only.

Optional research modes:
- structured answer mode,
- visible short reasoning mode.

Do not expose hidden reasoning by default.

## Dataset requirements
The data system must support iterable or streaming workflows.

Required behavior:
- Do not load the full dataset into RAM.
- Support local files and optional remote streaming sources.
- Read text incrementally.
- Tokenize incrementally.
- Assemble token blocks on the fly.
- Discard raw text after batch construction.

Implementation note:
- Prefer token-based chunking, not line-count rules.
- Support Hugging Face iterable/streaming datasets where useful.
- Keep train/validation behavior simple and explicit.

## Constitutional data schema
Support JSONL or equivalent records with fields such as:
- prompt
- draft
- critique
- revision
- final_answer
- short_reasoning
- principles_used

Template modes should include:
1. FINAL only
2. PLAN -> ANSWER
3. DRAFT -> CRITIQUE -> REVISED ANSWER
4. PRINCIPLES USED -> SHORT REASONING -> FINAL

Template selection must be configurable.
Default fine-tuning should bias the model toward final-answer-only inference.

## Hardware modes
The codebase must support multiple hardware modes.

### Mode A: Kaggle dual T4
- If two GPUs are available and training is launched in distributed mode, use DDP or Accelerate.
- Split or shard work across both GPUs.
- Use mixed precision where supported.
- Save checkpoints frequently.
- Keep resume behavior reliable.

### Mode B: Single GPU
- Support standard single-GPU training.
- Use gradient accumulation when needed.
- Use mixed precision where available.
- Keep memory usage stable.

### Mode C: Low-end CPU or older laptop
- Use tiny batch sizes.
- Use short sequence lengths.
- Keep RAM usage as low as practical, targeting under about 2 GB where possible.
- Disable expensive features when unsupported.
- Allow very small debug runs for pipeline validation.

## Codebase requirements
Code should be organized around a simple nanoGPT-like structure.

Preferred components:
- `model.py` for transformer model.
- `train.py` for base training.
- `finetune.py` for instruction/constitutional tuning.
- `sample.py` for inference.
- `data_utils.py` for streaming/iterable dataset logic.
- `config/` for model and hardware configs.
- `eval/` or `evaluate.py` for basic evaluation.

Guidelines:
- Prefer small explicit functions.
- Avoid giant abstractions.
- Avoid unnecessary dependencies.
- Add short docstrings to major modules.
- Keep changes minimal and easy to diff.

## Evaluation requirements
Include lightweight evaluation for:
- train/validation loss,
- approximate perplexity,
- instruction following,
- refusal style,
- concise structure adherence,
- reasoning-format adherence when enabled,
- low-memory safety in CPU mode.

Use small fixed prompt sets for regression testing.

## Gemini coding workflow
Gemini or another coding model may be used to write code, but it must be guided through narrowly scoped prompts.

Prompting rules:
- One task per prompt.
- Name exact files allowed to change.
- Name exact files that must not change.
- Ask for complete code for changed files.
- Ask for exact commands to run and test.
- Ask for minimal readable implementation.
- Do not ask for full-project rewrites.

Expected response format from coding model:
1. Short implementation plan.
2. Full code for changed files only.
3. Run/test commands.
4. Brief implementation notes.

## Suggested implementation order
1. Project scaffold.
2. Small config files.
3. Streaming dataset pipeline.
4. Single-GPU stable training.
5. Dual-GPU distributed support.
6. Fine-tuning entrypoint.
7. Constitutional template support.
8. Sampling improvements.
9. Evaluation scripts.

## Safety and realism constraints
Do not claim this model is equivalent to Claude or any frontier model.
Do not claim broad robust reasoning beyond the scale of the model.
Do not add misleading benchmark claims without evaluation.
Do not force hidden-thought tags in all outputs.
Do not implement unsafe hidden-reasoning exposure as default behavior.

## Definition of success
A successful first version should:
- train from scratch at small scale,
- run reliably on Kaggle or modest hardware,
- support instruction tuning,
- support synthetic constitutional fine-tuning,
- produce calm and structured final answers,
- optionally emit short visible reasoning in research mode,
- remain simple enough to understand and modify.
