# Model Guide for Codex

This folder owns the tiny causal LM architecture, generation loop, configuration object, and checkpoint IO.

## Big Picture

- `config.py` defines `TinyConfig`, including model width/depth, grouped-query attention settings, sliding-window schedule, RoPE theta, QK norm, PLE settings, and dropout.
- `attention.py` implements RMSNorm, RoPE, grouped-query key/value repetition, causal masks, local sliding-window attention, and global layer behavior.
- `model.py` builds `TinyGemmaLM` from token embeddings, optional per-layer embeddings, transformer blocks, final norm, and tied output head.
- `generation.py` performs simple autoregressive decoding with temperature, top-k filtering, EOS stopping, invalid-token suppression, and repetition penalty.
- `checkpoints.py` saves and loads model weights with `safetensors`.

## Important Contracts

- `cfg.d_model` must divide evenly by `cfg.n_heads`; `cfg.n_heads` must divide evenly by `cfg.n_kv_heads`.
- `TinyConfig.head_dim` is derived from `d_model // n_heads`.
- `TinyConfig.is_global_layer` makes every `local_layers_per_global` layer global and always makes the final layer global.
- Attention masks are boolean masks where `True` means the key position is visible.
- `TinyGemmaLM.forward` shifts logits and labels internally for causal LM loss and ignores `-100` labels.
- `lm_head.weight` is tied to `tok_embeddings.weight`; preserve this unless a task explicitly changes the architecture.
- `generate` recomputes over the cropped context instead of using a KV cache. Keep it simple unless the task is specifically about faster inference.

## Testing

Run model tests after changes here:

```bash
rtk uv run pytest tests/test_model.py
```

Also run the full suite when changing tensor shapes, loss computation, generation semantics, or checkpoint loading.

