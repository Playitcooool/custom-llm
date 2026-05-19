# Examples Guide for Codex

This folder contains tiny local inputs for README and CLI smoke runs.

## Big Picture

- `tiny.txt` is plain text for tokenizer training and pretraining smoke tests.
- `sft.jsonl` contains prompt/response rows for supervised fine-tuning.
- `prompts.txt` feeds the offline distillation path.

## Change Guidance

- Keep examples small, readable, and deterministic.
- Preserve JSONL shape for SFT examples: each non-empty line should contain `prompt` and `response`.
- Do not place generated checkpoints, tokenizers, downloaded corpora, or large datasets here.
- If README commands reference an example file, update both the example and README together.

