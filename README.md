# custom-llm

A small educational PyTorch repo for a text-only Gemma-inspired LLM. It is not weight or config compatible with Gemma; it mirrors selected ideas at laptop scale: RMSNorm, RoPE, QK norm, grouped-query attention, hybrid local/global attention, GeGLU-style MLPs, and optional per-layer embeddings.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run custom-llm train-tokenizer --files examples/tiny.txt --out .artifacts/tokenizer --vocab-size 128
uv run custom-llm pretrain --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --text examples/tiny.txt --out .artifacts/pretrain.safetensors
uv run custom-llm sft --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --jsonl examples/sft.jsonl --out .artifacts/sft.safetensors
uv run custom-llm distill --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --prompts examples/prompts.txt --out .artifacts/distill.safetensors
uv run custom-llm sample --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --checkpoint .artifacts/sft.safetensors --prompt "Tiny models"
```

## TinyStories Training

TinyStories is the recommended first real dataset for this model. Prepare a bounded local sample, train a tokenizer on it, then pretrain from the resulting text file:

```bash
uv run custom-llm prepare-tinystories --out data/tinystories_100mb.txt --max-mb 100
uv run custom-llm train-tokenizer --files data/tinystories_100mb.txt --out .artifacts/tokenizer --vocab-size 4096
uv run custom-llm pretrain --config configs/tiny.yaml --tokenizer .artifacts/tokenizer --text data/tinystories_100mb.txt --out .artifacts/tinystories.safetensors
uv run custom-llm sample --config configs/tiny.yaml --tokenizer .artifacts/tokenizer --checkpoint .artifacts/tinystories.safetensors --prompt "Once upon a time"
```

Start with `--max-mb 100` to validate the pipeline. On a 24 GB Mac, increase toward `500-2000` MB once the run is stable.

## Layout

- `custom_llm/model`: architecture, generation, and safetensors checkpoints
- `custom_llm/data`: tokenizer training, packing, JSONL SFT data, and collation
- `custom_llm/train`: pretraining, SFT, and offline distillation loops
- `custom_llm/teachers`: teacher interface and deterministic mock teacher
- `configs`: tiny and smoke YAML configs
- `tests`: architecture and workflow tests

The default distillation path is fully offline. It samples prompts from a local file, asks a deterministic mock teacher, and trains on those responses through the same SFT path.
