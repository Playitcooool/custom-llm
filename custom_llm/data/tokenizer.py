from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<user>", "<assistant>"]


def train_tokenizer(files: list[str], out_dir: str, vocab_size: int = 4096) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(files=files, vocab_size=vocab_size, min_frequency=1, special_tokens=SPECIAL_TOKENS)
    tok.save_model(out_dir)


def load_tokenizer(path: str) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        vocab = path.parent / "vocab.json"
        merges = path.parent / "merges.txt"
        tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        tok.save(str(path))
    return Tokenizer.from_file(str(path))


def encode(tokenizer: Tokenizer, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    ids = tokenizer.encode(text).ids
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if add_bos and bos is not None:
        ids = [bos] + ids
    if add_eos and eos is not None:
        ids = ids + [eos]
    return ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)
