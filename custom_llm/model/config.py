from dataclasses import dataclass


@dataclass
class TinyConfig:
    vocab_size: int = 4096
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 8
    n_kv_heads: int = 2
    intermediate_dim: int = 1536
    max_seq_len: int = 1024
    sliding_window: int = 128
    local_layers_per_global: int = 4
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    ple_dim: int = 64
    use_ple: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_global_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.local_layers_per_global == 0 or layer_idx == self.n_layers - 1
