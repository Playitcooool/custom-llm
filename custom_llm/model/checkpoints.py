from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from custom_llm.model.config import TinyConfig
from custom_llm.model.model import TinyGemmaLM


def save_checkpoint(model: TinyGemmaLM, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))


def load_checkpoint(path: str | Path, cfg: TinyConfig) -> TinyGemmaLM:
    model = TinyGemmaLM(cfg)
    load_model(model, str(path))
    return model
