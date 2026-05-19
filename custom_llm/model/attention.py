import math

import torch
from torch import nn
import torch.nn.functional as F

from custom_llm.model.config import TinyConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)


def build_causal_mask(seq_len: int, device: torch.device, sliding_window: int | None = None) -> torch.Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    mask = k <= q
    if sliding_window is not None:
        mask &= k >= (q - sliding_window + 1)
    return mask


def apply_rope(x: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    _, _, seq_len, head_dim = x.shape
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    t = torch.arange(seq_len, device=x.device).float()
    angles = torch.outer(t, freqs)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TinyConfig, layer_idx: int):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for grouped-query attention")
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_global = cfg.is_global_layer(layer_idx)
        self.n_kv_groups = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps) if cfg.qk_norm else nn.Identity()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), self.cfg.rope_theta)
        k = apply_rope(self.k_norm(k), self.cfg.rope_theta)
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)
        window = None if self.is_global else self.cfg.sliding_window
        mask = build_causal_mask(seq_len, x.device, window)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.cfg.head_dim)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = self.dropout(F.softmax(scores.float(), dim=-1).to(scores.dtype))
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.d_model)
        return self.o_proj(out)


Attention = GroupedQueryAttention
