import torch
from torch import nn
import torch.nn.functional as F

from custom_llm.model.attention import Attention, RMSNorm
from custom_llm.model.config import TinyConfig


class MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate_dim, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate_dim, bias=False)
        self.down = nn.Linear(cfg.intermediate_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig, layer_idx: int):
        super().__init__()
        self.input_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.post_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_norm(x))
        return x


class TinyGemmaLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ple = nn.ModuleList(
            [nn.Embedding(cfg.max_seq_len, cfg.ple_dim) for _ in range(cfg.n_layers)]
        ) if cfg.use_ple else None
        self.ple_proj = nn.ModuleList(
            [nn.Linear(cfg.ple_dim, cfg.d_model, bias=False) for _ in range(cfg.n_layers)]
        ) if cfg.use_ple else None
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        _, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        for i, layer in enumerate(self.layers):
            if self.ple is not None:
                x = x + self.ple_proj[i](self.ple[i](positions))[None, :, :]
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}
