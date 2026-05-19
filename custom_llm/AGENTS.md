# Package Guide for Codex

`custom_llm` is the importable package behind the `custom-llm` console script. The package is intentionally small; understand the control flow from `cli.py` before changing submodules.

## Control Flow

- `cli.py` parses commands and connects user inputs to data, model, and training modules.
- Tokenizer and dataset inputs come from `custom_llm.data`.
- Model instances are `custom_llm.model.model.TinyGemmaLM` configured by `TinyConfig`.
- Training commands call `custom_llm.train.pretrain.run_pretrain`, `custom_llm.train.sft.run_sft`, or `custom_llm.train.distill.run_distill`.
- Sampling loads a tokenizer, optionally loads a checkpoint, encodes the prompt, calls `generate`, and decodes only the generated suffix.

## Change Guidance

- Keep CLI defaults aligned with README examples.
- If adding a new command, add tests for argument-level helpers where practical and update the root README if it changes user workflow.
- Import across package boundaries in the same direct style already used here; avoid introducing registries or framework layers.
- Keep optional dependencies out of import-time paths unless they are already required by `pyproject.toml`.

