from custom_llm.cli import checkpoint_output_path
from custom_llm.train.utils import resolve_device


def test_checkpoint_output_defaults_use_stage_extensions():
    assert checkpoint_output_path("pretrain", None) == ".artifacts/pretrain.safetensors"
    assert checkpoint_output_path("sft", None) == ".artifacts/sft.safetensors"
    assert checkpoint_output_path("distill", None) == ".artifacts/distill.safetensors"


def test_checkpoint_output_normalizes_known_safetensor_suffixes():
    assert checkpoint_output_path("pretrain", "runs/model.safetensor") == "runs/model.safetensors"
    assert checkpoint_output_path("sft", "runs/model.safetensor") == "runs/model.safetensors"
    assert checkpoint_output_path("distill", "runs/model.safetensor") == "runs/model.safetensors"


def test_checkpoint_output_appends_suffix_only_when_missing():
    assert checkpoint_output_path("pretrain", "runs/model") == "runs/model.safetensors"
    assert checkpoint_output_path("sft", "runs/model") == "runs/model.safetensors"
    assert checkpoint_output_path("pretrain", "runs/model.bin") == "runs/model.bin"


def test_resolve_device_accepts_auto_and_cpu():
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
    assert resolve_device("cpu").type == "cpu"
