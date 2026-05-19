# Repository Guide for Codex

Read `/Users/weiciruan/.codex/RTK.md` before running shell commands. In this repo, prefer `rtk <command>` for command execution.

## Big Picture

This is a small educational PyTorch implementation of a text-only, Gemma-inspired language model. It is not compatible with Gemma weights or configs. The repo exists to keep the full LLM workflow understandable at laptop scale:

1. Prepare or sample text data.
2. Train a byte-level BPE tokenizer.
3. Build a tiny causal LM with RMSNorm, RoPE, grouped-query attention, local/global attention, GeGLU-style MLPs, and optional per-layer embeddings.
4. Run pretraining, supervised fine-tuning, or mock-teacher distillation.
5. Save checkpoints with `safetensors` and sample text through the CLI.

The CLI in `custom_llm/cli.py` is the main integration point. Most user-facing flows should remain reachable through `uv run custom-llm ...`.

## Main Areas

- `custom_llm/model`: model architecture, attention, generation, config, and checkpoint IO.
- `custom_llm/data`: tokenizer training/loading, text packing, SFT JSONL parsing, collation, and dataset preparation.
- `custom_llm/train`: pretraining, SFT, distillation orchestration, device selection, config parsing, and optimization steps.
- `custom_llm/teachers`: teacher interface plus deterministic offline mock teacher.
- `configs`: YAML model and training presets. `smoke.yaml` is for fast tests; `tiny.yaml` is the realistic local baseline.
- `examples`: tiny local inputs for CLI smoke runs.
- `tests`: pytest coverage for architecture, data handling, training smoke paths, and CLI helpers.

## Development Workflow

- Use Python 3.10 or 3.11 as required by `pyproject.toml`.
- Install and verify with:

```bash
rtk uv sync --extra dev
rtk uv run pytest
```

- For quick end-to-end validation, use `configs/smoke.yaml` and the files in `examples/`.
- Keep generated data, tokenizers, and checkpoints in `.artifacts/` or another ignored scratch path. Do not commit model artifacts.
- Use `safetensors` for checkpoints unless a task explicitly asks for another format.

## Design Constraints

- Preserve the repository's educational clarity. Prefer explicit PyTorch code over clever abstractions.
- Keep public behavior wired through the CLI when adding new workflows.
- Keep config keys compatible with `custom_llm.model.config.TinyConfig` and `custom_llm.train.utils.train_cfg`.
- Avoid adding online service dependencies to the default path. The default distillation workflow must remain offline and deterministic.
- When changing tensor shapes, masking, loss shifting, tokenizer special tokens, or checkpoint loading, add or update tests.

## Common Commands

```bash
rtk uv run pytest
rtk uv run pytest tests/test_model.py
rtk uv run custom-llm train-tokenizer --files examples/tiny.txt --out .artifacts/tokenizer --vocab-size 128
rtk uv run custom-llm pretrain --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --text examples/tiny.txt --out .artifacts/pretrain.safetensors
rtk uv run custom-llm sample --config configs/smoke.yaml --tokenizer .artifacts/tokenizer --checkpoint .artifacts/pretrain.safetensors --prompt "Tiny models"
```

