from pathlib import Path
from typing import Iterable


FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"


def write_text_rows(
    rows: Iterable[dict],
    out_file: str | Path,
    max_bytes: int,
    text_field: str = "text",
    min_chars: int = 200,
    max_docs: int | None = None,
) -> Path:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    docs = 0
    with out_path.open("wb") as out:
        for row in rows:
            text = row.get(text_field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < min_chars:
                continue
            data = (text + "\n\n").encode("utf-8")
            remaining = max_bytes - written
            if remaining <= 0:
                break
            out.write(data[:remaining])
            written += min(len(data), remaining)
            docs += 1
            if max_docs is not None and docs >= max_docs:
                break
    if written == 0:
        raise ValueError(f"no usable text rows found for field {text_field!r}")
    return out_path


def prepare_fineweb_edu(
    out_file: str,
    max_mb: int = 100,
    dataset: str = FINEWEB_EDU_DATASET,
    name: str = FINEWEB_EDU_CONFIG,
    split: str = "train",
    text_field: str = "text",
    min_chars: int = 200,
    max_docs: int | None = None,
) -> Path:
    from datasets import load_dataset

    rows = load_dataset(dataset, name=name, split=split, streaming=True)
    return write_text_rows(
        rows=rows,
        out_file=out_file,
        max_bytes=max_mb * 1024 * 1024,
        text_field=text_field,
        min_chars=min_chars,
        max_docs=max_docs,
    )
