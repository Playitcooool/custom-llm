import argparse
from pathlib import Path

import torch

from custom_llm.data.tokenizer import (
    decode,
    encode,
    invalid_single_token_ids,
    load_tokenizer,
    train_tokenizer,
)
from custom_llm.data.fineweb_edu import (
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    prepare_fineweb_edu,
)
from custom_llm.data.tinystories import TINYSTORIES_URL, prepare_tinystories
from custom_llm.model.checkpoints import load_checkpoint
from custom_llm.model.generation import generate
from custom_llm.model.model import TinyGemmaLM
from custom_llm.train.distill import run_distill
from custom_llm.train.pretrain import run_pretrain
from custom_llm.train.sft import run_sft
from custom_llm.train.utils import load_config, resolve_device, tiny_config_from_dict

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

    fineweb = sub.add_parser("prepare-fineweb-edu")
    fineweb.add_argument("--out", default="data/fineweb_edu_100mb.txt")
    fineweb.add_argument("--max-mb", type=int, default=100)
    fineweb.add_argument("--dataset", default=FINEWEB_EDU_DATASET)
    fineweb.add_argument("--name", default=FINEWEB_EDU_CONFIG)
    fineweb.add_argument("--split", default="train")
    fineweb.add_argument("--text-field", default="text")
    fineweb.add_argument("--min-chars", type=int, default=200)
    fineweb.add_argument("--max-docs", type=int, default=None)

    for name in ("pretrain", "sft", "distill"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/tiny.yaml")
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--out", default=None)
        p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
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
    sample.add_argument("--temperature", type=float, default=0.0)
    sample.add_argument("--top-k", type=int, default=50)
    sample.add_argument("--repetition-penalty", type=float, default=1.1)
    sample.add_argument("--allow-invalid-bytes", action="store_true")
    sample.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))

    args = parser.parse_args()
    if args.cmd == "train-tokenizer":
        train_tokenizer(args.files, args.out, args.vocab_size)
    elif args.cmd == "prepare-tinystories":
        out = prepare_tinystories(args.out, args.cache_dir, args.max_mb, args.url)
        print(out)
    elif args.cmd == "prepare-fineweb-edu":
        out = prepare_fineweb_edu(
            out_file=args.out,
            max_mb=args.max_mb,
            dataset=args.dataset,
            name=args.name,
            split=args.split,
            text_field=args.text_field,
            min_chars=args.min_chars,
            max_docs=args.max_docs,
        )
        print(out)
    elif args.cmd == "pretrain":
        run_pretrain(
            load_config(args.config),
            args.text,
            args.tokenizer,
            checkpoint_output_path(args.cmd, args.out),
            args.device,
        )
    elif args.cmd == "sft":
        run_sft(
            load_config(args.config),
            args.jsonl,
            args.tokenizer,
            checkpoint_output_path(args.cmd, args.out),
            args.device,
        )
    elif args.cmd == "distill":
        run_distill(
            load_config(args.config),
            args.prompts,
            args.tokenizer,
            checkpoint_output_path(args.cmd, args.out),
            args.device,
        )
    elif args.cmd == "sample":
        cfg = tiny_config_from_dict(load_config(args.config))
        tokenizer = load_tokenizer(args.tokenizer)
        cfg.vocab_size = max(cfg.vocab_size, tokenizer.get_vocab_size())
        dev = resolve_device(args.device)
        print(f"device: {dev}")
        model = load_checkpoint(args.checkpoint, cfg) if args.checkpoint else TinyGemmaLM(cfg)
        model = model.to(dev)
        prompt_ids = encode(tokenizer, args.prompt, add_eos=False)
        ids = torch.tensor([prompt_ids], device=dev)
        eos_id = tokenizer.token_to_id("<eos>")
        suppress_ids = None if args.allow_invalid_bytes else invalid_single_token_ids(tokenizer)
        out = generate(
            model,
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
            suppress_token_ids=suppress_ids,
            repetition_penalty=args.repetition_penalty,
        )
        generated_ids = out[0, len(prompt_ids) :].tolist()
        print(decode(tokenizer, generated_ids))


if __name__ == "__main__":
    main()
