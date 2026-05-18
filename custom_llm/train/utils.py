from dataclasses import fields
from pathlib import Path

import torch
import yaml

from custom_llm.model.config import TinyConfig


def load_config(path: str | None) -> dict:
    return yaml.safe_load(Path(path).read_text()) if path else {}


def tiny_config_from_dict(data: dict) -> TinyConfig:
    model = data.get("model", data)
    allowed = {f.name for f in fields(TinyConfig)}
    return TinyConfig(**{k: v for k, v in model.items() if k in allowed})


def train_cfg(data: dict) -> dict:
    return data.get("train", {})


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimization_step(model, batch, optimizer, dev):
    batch = {k: v.to(dev) for k, v in batch.items()}
    out = model(batch["input_ids"], batch["labels"])
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(out["loss"].detach().cpu())
