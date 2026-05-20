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

Training and sampling use `--device auto` by default. Auto picks CUDA first, then Apple MPS, then CPU. Override it with `--device mps` or `--device cpu` when needed.

AdamW is the default optimizer. To try Muon for hidden 2D weights while keeping AdamW for embeddings, norms, and other fallback parameters, set it in the config:

```yaml
train:
  optimizer: muon
  lr: 0.0003
  adamw_lr: 0.0003
  muon_lr: 0.02
  muon_momentum: 0.95
  muon_ns_steps: 5
```

## Config Reference

Configs have two main sections: `model`, which builds the network, and `train`, which controls the training loop. See `configs/smoke.yaml` for a fast test preset and `configs/tiny.yaml` for a more realistic local baseline.

### Model parameters

- `vocab_size`: Number of token IDs the model can represent. Training raises this to at least the loaded tokenizer size when needed.
- `n_layers`: Number of Transformer blocks.
- `d_model`: Hidden width of the model. Token embeddings, attention outputs, residual connections, and final normalization all use this size.
- `n_heads`: Number of query attention heads.
- `n_kv_heads`: Number of key/value heads for grouped-query attention. `n_heads` must be divisible by `n_kv_heads`.
- `intermediate_dim`: Inner width of the GeGLU-style MLP. Larger values increase capacity and compute.
- `max_seq_len`: Maximum sequence length accepted by the model. Inputs longer than this raise an error.
- `sliding_window`: Local attention window size for non-global layers. Global layers still attend to the full causal prefix.
- `local_layers_per_global`: Frequency of global attention layers. Every Nth layer is global, and the final layer is always global.
- `rope_theta`: Base frequency scale for RoPE positional encoding.
- `rms_norm_eps`: Epsilon used by RMSNorm for numerical stability.
- `qk_norm`: Enables RMSNorm on query and key vectors before RoPE and attention.
- `ple_dim`: Width of the per-layer learned positional embedding before projection to `d_model`.
- `use_ple`: Enables per-layer learned positional embeddings.
- `dropout`: Dropout probability applied to attention probabilities.

Derived constraint: `head_dim = d_model // n_heads`, so `d_model` must be divisible by `n_heads`.

### Training parameters

- `seq_len`: Sequence length used by the dataset packer. This should usually be less than or equal to `model.max_seq_len`.
- `batch_size`: Number of sequences per optimizer step.
- `steps`: Number of training iterations.
- `lr`: Main learning rate. AdamW uses this directly; Muon uses it as the fallback AdamW learning rate unless `adamw_lr` is set.
- `optimizer`: Optimizer name. Supported values are `adamw` and `muon`; default is `adamw`.
- `weight_decay`: Weight decay for optimizer parameter groups. Defaults to `0.0`.
- `adamw_lr`: Learning rate for AdamW parameters when using `optimizer: muon`.
- `muon_lr`: Learning rate for 2D hidden-weight parameters trained with Muon.
- `muon_momentum`: Momentum value for Muon.
- `muon_ns_steps`: Number of Newton-Schulz orthogonalization steps used by Muon.
- `muon_nesterov`: Enables Nesterov-style Muon momentum. Defaults to `true`.

Practical rule: `model.max_seq_len` is the architectural limit, while `train.seq_len` is how much of that limit each training example uses.

## FineWeb-Edu Training

FineWeb-Edu is the recommended first realistic dataset for this model. It is educational web text filtered from FineWeb and is much closer to real pretraining data than TinyStories. Prepare a bounded local sample, train a tokenizer on it, then pretrain from the resulting text file:

```bash
uv run custom-llm prepare-fineweb-edu --out data/fineweb_edu_100mb.txt --max-mb 100
uv run custom-llm train-tokenizer --files data/fineweb_edu_100mb.txt --out .artifacts/tokenizer --vocab-size 4096
uv run custom-llm pretrain --config configs/tiny.yaml --tokenizer .artifacts/tokenizer --text data/fineweb_edu_100mb.txt --out .artifacts/fineweb_edu.safetensors
uv run custom-llm sample --config configs/tiny.yaml --tokenizer .artifacts/tokenizer --checkpoint .artifacts/fineweb_edu.safetensors --prompt "Explain photosynthesis in simple terms:"
```

Restart pretraining from an existing model checkpoint by passing `--restart-checkpoint`:

```bash
uv run custom-llm pretrain --config configs/tiny.yaml --tokenizer .artifacts/tokenizer --text data/fineweb_edu_100mb.txt --restart-checkpoint .artifacts/fineweb_edu.safetensors --out .artifacts/fineweb_edu_next.safetensors
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
