import argparse
from pathlib import Path

import torch

from custom_llm.data.tokenizer import decode, encode, load_tokenizer, train_tokenizer
from custom_llm.data.tinystories import TINYSTORIES_URL, prepare_tinystories
from custom_llm.model.checkpoints import load_checkpoint
from custom_llm.model.generation import generate
from custom_llm.model.model import TinyGemmaLM
from custom_llm.train.distill import run_distill
from custom_llm.train.pretrain import run_pretrain
from custom_llm.train.sft import run_sft
from custom_llm.train.utils import load_config, tiny_config_from_dict

CHECKPOINT_EXTENSIONS = {
    "pretrain": ".safetensors",
    "sft": ".safetensors",
    "distill": ".safetensors",
}


def checkpoint_output_path(stage: str, out: str | None) -> str:
    expected = CHECKPOINT_EXTENSIONS[stage]
    path = Path(out) if out else Path(".artifacts") / f"{stage}{expected}"
    if path.suffix in {".safetensor", ".safetensors"} and path.suffix != expected:
        path = path.with_suffix(expected)
    elif path.suffix == "":
        path = path.with_suffix(expected)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="custom-llm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tok = sub.add_parser("train-tokenizer")
    tok.add_argument("--files", nargs="+", required=True)
    tok.add_argument("--out", required=True)
    tok.add_argument("--vocab-size", type=int, default=4096)

    tiny = sub.add_parser("prepare-tinystories")
    tiny.add_argument("--out", default="data/tinystories_sample.txt")
    tiny.add_argument("--cache-dir", default=".artifacts/downloads")
    tiny.add_argument("--max-mb", type=int, default=100)
    tiny.add_argument("--url", default=TINYSTORIES_URL)

    for name in ("pretrain", "sft", "distill"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/tiny.yaml")
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--out", default=None)
        if name == "pretrain":
            p.add_argument("--text", nargs="+", required=True)
        elif name == "sft":
            p.add_argument("--jsonl", required=True)
        else:
            p.add_argument("--prompts", required=True)

    sample = sub.add_parser("sample")
    sample.add_argument("--config", default="configs/tiny.yaml")
    sample.add_argument("--tokenizer", required=True)
    sample.add_argument("--checkpoint", default=None)
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--max-new-tokens", type=int, default=32)

    args = parser.parse_args()
    if args.cmd == "train-tokenizer":
        train_tokenizer(args.files, args.out, args.vocab_size)
    elif args.cmd == "prepare-tinystories":
        out = prepare_tinystories(args.out, args.cache_dir, args.max_mb, args.url)
        print(out)
    elif args.cmd == "pretrain":
        run_pretrain(load_config(args.config), args.text, args.tokenizer, checkpoint_output_path(args.cmd, args.out))
    elif args.cmd == "sft":
        run_sft(load_config(args.config), args.jsonl, args.tokenizer, checkpoint_output_path(args.cmd, args.out))
    elif args.cmd == "distill":
        run_distill(
            load_config(args.config),
            args.prompts,
            args.tokenizer,
            checkpoint_output_path(args.cmd, args.out),
        )
    elif args.cmd == "sample":
        cfg = tiny_config_from_dict(load_config(args.config))
        tokenizer = load_tokenizer(args.tokenizer)
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.get_vocab_size())
        model = load_checkpoint(args.checkpoint, cfg) if args.checkpoint else TinyGemmaLM(cfg)
        ids = torch.tensor([encode(tokenizer, args.prompt, add_eos=False)])
        out = generate(model, ids, args.max_new_tokens)
        print(decode(tokenizer, out[0].tolist()))


if __name__ == "__main__":
    main()
