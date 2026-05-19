import torch

from custom_llm.model.attention import GroupedQueryAttention, build_causal_mask, repeat_kv
from custom_llm.model.config import TinyConfig
from custom_llm.model.model import TinyGemmaLM


def small_cfg(**kwargs):
    data = dict(
        vocab_size=128,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_dim=64,
        max_seq_len=32,
        sliding_window=3,
        local_layers_per_global=2,
        ple_dim=8,
    )
    data.update(kwargs)
    return TinyConfig(**data)


def test_causal_mask_blocks_future():
    mask = build_causal_mask(4, torch.device("cpu"))
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_sliding_window_mask_limits_history():
    mask = build_causal_mask(5, torch.device("cpu"), sliding_window=2)
    assert mask[4].tolist() == [False, False, False, True, True]


def test_global_layer_schedule_forces_final_global():
    cfg = small_cfg(n_layers=5, local_layers_per_global=4)
    assert not cfg.is_global_layer(0)
    assert cfg.is_global_layer(3)
    assert cfg.is_global_layer(4)


def test_gqa_kv_repetition():
    x = torch.arange(1 * 2 * 3 * 4).view(1, 2, 3, 4)
    y = repeat_kv(x, 2)
    assert y.shape == (1, 4, 3, 4)
    assert torch.equal(y[:, 0], x[:, 0])
    assert torch.equal(y[:, 1], x[:, 0])
    assert torch.equal(y[:, 2], x[:, 1])


def test_grouped_query_attention_uses_fewer_kv_heads():
    cfg = small_cfg(n_heads=4, n_kv_heads=2)
    attn = GroupedQueryAttention(cfg, layer_idx=0)
    assert attn.n_kv_groups == 2
    assert attn.q_proj.out_features == cfg.n_heads * cfg.head_dim
    assert attn.k_proj.out_features == cfg.n_kv_heads * cfg.head_dim
    assert attn.v_proj.out_features == cfg.n_kv_heads * cfg.head_dim
    assert attn.k_proj.out_features < attn.q_proj.out_features


def test_model_forward_and_ple_toggle():
    cfg = small_cfg(use_ple=True)
    model = TinyGemmaLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x, x)
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert out["loss"].ndim == 0
    no_ple = TinyGemmaLM(small_cfg(use_ple=False))
    assert no_ple.ple is None
