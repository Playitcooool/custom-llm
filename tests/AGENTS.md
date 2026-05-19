# Tests Guide for Codex

This folder contains pytest coverage for the educational LLM workflow.

## Big Picture

- `test_model.py` covers attention masks, global-layer scheduling, grouped-query repetition, model forward pass, PLE toggling, and generation controls.
- `test_data_and_train.py` covers tokenizer training, dataset preparation, SFT masking, collation, and one-step pretrain/SFT/distill paths.
- `test_cli.py` covers checkpoint path normalization and device selection helpers.

## Change Guidance

- Prefer small CPU-friendly tests with temporary files.
- Keep tests deterministic; avoid live downloads or external API calls.
- For data prep that normally streams remote data, test pure helper functions with local fixtures.
- When adding a CLI-visible behavior, test the helper or a narrow invocation path instead of running a long training job.

## Commands

```bash
rtk uv run pytest
rtk uv run pytest tests/test_model.py
rtk uv run pytest tests/test_data_and_train.py
rtk uv run pytest tests/test_cli.py
```

