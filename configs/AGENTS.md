# Configs Guide for Codex

This folder contains YAML presets consumed by `custom_llm.train.utils.load_config`.

## Big Picture

- `smoke.yaml` is for fast local and CI-style validation.
- `tiny.yaml` is the more realistic laptop-scale baseline.
- The `model` section maps onto fields in `custom_llm.model.config.TinyConfig`.
- The `train` section is read as a plain dict by `custom_llm.train.utils.train_cfg`.

## Change Guidance

- Keep `smoke.yaml` tiny enough for `rtk uv run pytest` and quick CLI smoke runs.
- Do not add config keys expecting automatic validation unless you also update `train/utils.py` or the relevant caller.
- If adding a model field to `TinyConfig`, update configs and tests together.
- Prefer explicit numeric values over computed or environment-dependent settings.

