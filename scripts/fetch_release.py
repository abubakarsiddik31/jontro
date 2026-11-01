"""Download the public Jontro 2.0 release and lay it out where the code expects it.

The scripts and tests in this repository read the corpus, the goal-disjoint splits
and the Parquet files from ``outputs/``. This script fills that directory from the
Hugging Face release, so nothing needs to be regenerated:

    python scripts/fetch_release.py

Files are verified against the release's ``SHA256SUMS``.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "abubakar-siddik/Jontro"
OUT = Path(__file__).resolve().parent.parent / "outputs"

# (file in the Hugging Face repo, destination under outputs/)
FILES = [
    ("raw/jontro_multiturn_2000.jsonl", "release_multiturn/jontro_multiturn_2000.jsonl"),
    ("raw/jontro_multiturn_2000.jsonl", "dataset/bangla_agentic.jsonl"),
    ("raw/jontro_multiturn_2000_openai.jsonl", "release_multiturn/jontro_multiturn_2000_openai.jsonl"),
    ("raw/jontro_multiturn_2000_sharegpt.jsonl", "release_multiturn/jontro_multiturn_2000_sharegpt.jsonl"),
    ("raw/jontro_multiturn_100_golden.jsonl", "release_multiturn/jontro_multiturn_100.jsonl"),
    ("splits/train_ids.jsonl", "splits/train_ids.jsonl"),
    ("splits/test_ids.jsonl", "splits/test_ids.jsonl"),
    ("splits/test_goals.jsonl", "splits/test_goals.jsonl"),
    ("data/train/jontro_train.parquet", "parquet/jontro_train.parquet"),
    ("data/test/jontro_test.parquet", "parquet/jontro_test.parquet"),
    ("data/train/jontro_train_sharegpt.parquet", "parquet/jontro_train_sharegpt.parquet"),
    ("data/test/jontro_test_sharegpt.parquet", "parquet/jontro_test_sharegpt.parquet"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sums = {}
    for line in Path(hf_hub_download(REPO, "SHA256SUMS", repo_type="dataset")).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0]
    for remote, dest in FILES:
        src = Path(hf_hub_download(REPO, remote, repo_type="dataset"))
        expected = sums.get(remote)
        if expected and sha256(src) != expected:
            raise SystemExit(f"checksum mismatch for {remote}")
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        print(f"{remote} -> outputs/{dest}")


if __name__ == "__main__":
    main()
