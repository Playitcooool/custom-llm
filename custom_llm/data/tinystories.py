import json
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStories_all_data.tar.gz"
)

GZIP_MAGIC = b"\x1f\x8b"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as out:
        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total or None, unit="B", unit_scale=True, desc="download") as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def is_gzip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2:
        return False
    with path.open("rb") as f:
        return f.read(2) == GZIP_MAGIC


def _write_text(out, text: str, written: int, max_bytes: int) -> int:
    data = (text.strip() + "\n\n").encode("utf-8")
    remaining = max_bytes - written
    if remaining <= 0:
        return written
    out.write(data[:remaining])
    return written + min(len(data), remaining)


def _copy_member_text(source, out, written: int, max_bytes: int) -> int:
    while written < max_bytes:
        chunk = source.read(min(1024 * 1024, max_bytes - written))
        if not chunk:
            break
        out.write(chunk)
        written += len(chunk)
    return written


def _extract_from_tar(tar: tarfile.TarFile, out, max_bytes: int) -> int:
    written = 0
    for member in tar:
        if not member.isfile():
            continue
        source = tar.extractfile(member)
        if source is None:
            continue
        if member.name.endswith(".txt"):
            written = _copy_member_text(source, out, written, max_bytes)
        elif member.name.endswith(".json"):
            rows = json.load(source)
            for row in rows:
                story = row.get("story") if isinstance(row, dict) else None
                if story:
                    written = _write_text(out, story, written, max_bytes)
                if written >= max_bytes:
                    break
        if written >= max_bytes:
            break
    return written


def extract_text_sample(archive: Path, out_file: Path, max_bytes: int) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar, out_file.open("wb") as out:
        written = _extract_from_tar(tar, out, max_bytes)
    if written == 0:
        raise ValueError(f"no TinyStories text found in {archive}")


def stream_text_sample(url: str, out_file: Path, max_bytes: int) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        with out_file.open("wb") as out:
            written = _extract_from_tar(tar, out, max_bytes)
    if written == 0:
        raise ValueError(f"no TinyStories text found in {url}")


def prepare_tinystories(
    out_file: str,
    cache_dir: str = ".artifacts/downloads",
    max_mb: int = 100,
    url: str = TINYSTORIES_URL,
) -> Path:
    archive = Path(cache_dir) / "TinyStories_all_data.tar.gz"
    max_bytes = max_mb * 1024 * 1024
    out_path = Path(out_file)
    if archive.exists() and not is_gzip(archive):
        archive.unlink()
    if not archive.exists():
        stream_text_sample(url, out_path, max_bytes)
    else:
        extract_text_sample(archive, out_path, max_bytes)
    return out_path
