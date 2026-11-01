"""Canonical JSONL writer (one trajectory per line) + verdict writer."""
from __future__ import annotations

from pathlib import Path

from bangla_datasets.schema import Trajectory, Verdict


def write_jsonl(trajectories: list[Trajectory], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in trajectories:
            f.write(t.model_dump_json(ensure_ascii=False) + "\n")


def write_verdicts(verdicts: list[Verdict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in verdicts:
            f.write(v.model_dump_json(ensure_ascii=False) + "\n")
