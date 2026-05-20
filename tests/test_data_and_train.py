import json
import tarfile

import torch

from custom_llm.data.collate import causal_lm_collate
from custom_llm.data.datasets import SFTDataset
from custom_llm.data.fineweb_edu import write_text_rows
from custom_llm.data.tinystories import extract_text_sample, is_gzip
from custom_llm.data.tokenizer import load_tokenizer, train_tokenizer
from custom_llm.model.checkpoints import load_checkpoint
from custom_llm.train.distill import run_distill
from custom_llm.train.pretrain import run_pretrain
from custom_llm.train.sft import run_sft
from custom_llm.train.utils import tiny_config_from_dict


def smoke_config(vocab_size=128):
    return {
        "model": {
            "vocab_size": vocab_size,
            "n_layers": 1,
            "d_model": 24,
            "n_heads": 4,
            "n_kv_heads": 2,
            "intermediate_dim": 48,
            "max_seq_len": 24,
            "sliding_window": 8,
            "local_layers_per_global": 2,
            "ple_dim": 4,
            "use_ple": True,
        },
        "train": {"seq_len": 12, "batch_size": 1, "steps": 1, "lr": 1e-3},
    }


def make_tokenizer(tmp_path):
    text = tmp_path / "tiny.txt"
    text.write_text("hello tiny model\nlocal attention and global attention\n")
    out = tmp_path / "tok"
    train_tokenizer([str(text)], str(out), vocab_size=128)
    return text, out, load_tokenizer(str(out))


def test_tokenizer_training(tmp_path):
    _, out, tok = make_tokenizer(tmp_path)
    assert (out / "vocab.json").exists()
    assert tok.token_to_id("<bos>") is not None


def test_tinystories_extracts_bounded_text_sample(tmp_path):
    source = tmp_path / "part.txt"
    source.write_text("Once upon a time.\n" * 20)
    archive = tmp_path / "tiny.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="TinyStories/part.txt")
    out = tmp_path / "sample.txt"
    assert is_gzip(archive)
    extract_text_sample(archive, out, max_bytes=64)
    text = out.read_text()
    assert "Once upon a time." in text
    assert out.stat().st_size <= 65


def test_tinystories_rejects_lfs_pointer(tmp_path):
    pointer = tmp_path / "pointer.tar.gz"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    assert not is_gzip(pointer)


def test_fineweb_edu_writes_bounded_text_sample(tmp_path):
    rows = [
        {"text": "short"},
        {"text": "Photosynthesis turns sunlight into plant energy. " * 20},
        {"text": "A second educational web document explains gravity. " * 20},
    ]
    out = write_text_rows(rows, tmp_path / "fineweb.txt", max_bytes=256, min_chars=100)
    text = out.read_text()
    assert "Photosynthesis" in text
    assert "short" not in text
    assert out.stat().st_size == 256


def test_sft_masks_prompt_loss(tmp_path):
    _, _, tok = make_tokenizer(tmp_path)
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"prompt": "hello", "response": "tiny model"}) + "\n")
    ds = SFTDataset(str(path), tok, seq_len=24)
    row = ds[0]
    assert (row["labels"] == -100).any()
    assert (row["labels"] != -100).any()
    batch = causal_lm_collate([row], pad_id=0)
    assert batch["input_ids"].shape == batch["labels"].shape


def test_one_step_pretrain_sft_distill(tmp_path):
    text, tok_dir, _ = make_tokenizer(tmp_path)
    cfg = smoke_config()
    run_pretrain(cfg, [str(text)], str(tok_dir))
    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(json.dumps({"prompt": "hello", "response": "tiny model"}) + "\n")
    run_sft(cfg, str(sft_path), str(tok_dir))
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello tiny model\n")
    run_distill(cfg, str(prompts), str(tok_dir))


def test_pretrain_can_restart_from_checkpoint(tmp_path):
    text, tok_dir, tok = make_tokenizer(tmp_path)
    cfg = smoke_config(vocab_size=tok.get_vocab_size())
    first = tmp_path / "first.safetensors"
    restarted = tmp_path / "restarted.safetensors"

    run_pretrain(cfg, [str(text)], str(tok_dir), out=str(first), device_name="cpu")
    no_step_cfg = smoke_config(vocab_size=tok.get_vocab_size())
    no_step_cfg["train"]["steps"] = 0
    run_pretrain(
        no_step_cfg,
        [str(text)],
        str(tok_dir),
        out=str(restarted),
        device_name="cpu",
        restart_checkpoint=str(first),
    )

    model_cfg = tiny_config_from_dict(cfg)
    original = load_checkpoint(first, model_cfg)
    resumed = load_checkpoint(restarted, model_cfg)
    for key, value in original.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])


def test_loss_accepts_masked_labels():
    labels = torch.tensor([[1, -100, 3]])
    assert labels.shape == (1, 3)
