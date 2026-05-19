# Training Guide for Codex

This folder owns training orchestration for pretraining, supervised fine-tuning, and offline mock-teacher distillation.

## Big Picture

- `utils.py` loads YAML config dictionaries, extracts `TinyConfig` fields, resolves devices, and performs one optimization step.
- `pretrain.py` trains `TinyGemmaLM` on packed plain-text blocks.
- `sft.py` trains `TinyGemmaLM` on JSONL prompt/response examples.
- `distill.py` builds temporary SFT data from prompts using `MockTeacher`, then delegates to the SFT path.

## Important Contracts

- Device selection is `cuda`, then Apple `mps`, then `cpu` for `--device auto`.
- Training config lives under the YAML `train` key; model config lives under `model`.
- Tokenizer vocab size can increase `cfg.vocab_size` before model creation.
- Pretraining and SFT should share the same `optimization_step` and collator behavior unless a task intentionally changes the loss path.
- Distillation should clean up `.distill_teacher_data.jsonl` even on failure.
- Keep default runs small enough for `configs/smoke.yaml` to finish quickly in tests.

## Testing

Run:

```bash
rtk uv run pytest tests/test_data_and_train.py tests/test_cli.py
```

When changing device resolution, config parsing, checkpoint output naming, or command behavior, include CLI-related tests.

