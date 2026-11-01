"""Write all four Parquet split files (train/test x canonical/sharegpt).

Usage:
    python scripts/write_parquet.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import write_split_parquet

DATASET = Path("outputs/dataset/bangla_agentic.jsonl")
OUT = Path("outputs/parquet")


def _load_ids(path: Path) -> set[str]:
    return {json.loads(l)["id"] for l in path.read_text().splitlines()}


def main() -> None:
    trajs = [
        Trajectory.model_validate(json.loads(l))
        for l in DATASET.read_text().splitlines()
    ]
    train_ids = _load_ids(Path("outputs/splits/train_ids.jsonl"))
    test_ids = _load_ids(Path("outputs/splits/test_ids.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_split_parquet(trajs, train_ids, OUT / "jontro_train.parquet", fmt="canonical")
    write_split_parquet(trajs, test_ids, OUT / "jontro_test.parquet", fmt="canonical")
    write_split_parquet(trajs, train_ids, OUT / "jontro_train_sharegpt.parquet", fmt="sharegpt")
    write_split_parquet(trajs, test_ids, OUT / "jontro_test_sharegpt.parquet", fmt="sharegpt")
    for p in sorted(OUT.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
