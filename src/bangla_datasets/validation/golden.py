"""Golden set loader + judge drift detection.

Loads 15 hand-authored trajectories and their expected pass/fail labels. The CLI
(Task 10) scores these against the Gemini judge every run; if agreement with the
hand-labels drops below 85%, the run pauses — a hedge against the judge silently
degrading.

Path note: `golden.py` lives at `<repo>/src/bangla_datasets/validation/golden.py`
and the data lives at `<repo>/validation/golden/`. Counting `.resolve().parents`:
parents[0]=validation, parents[1]=bangla_datasets, parents[2]=src, parents[3]=repo
root. So the data dir is `parents[3] / "validation" / "golden"`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bangla_datasets.schema import Trajectory, Verdict

# parents[3] = repo root (see module docstring). The brief used parents[2],
# which resolves to <repo>/src/validation/golden — an off-by-one we correct here.
_GOLDEN_DIR = Path(__file__).resolve().parents[3] / "validation" / "golden"


@dataclass
class DriftReport:
    agreement_rate: float
    disagreements: list[str]


class GoldenSet:
    """Loads hand-written trajectories + labels for judge-drift detection."""

    def __init__(self, golden_dir: Path = _GOLDEN_DIR) -> None:
        self._dir = golden_dir

    def load(self) -> list[Trajectory]:
        out: list[Trajectory] = []
        for line in (self._dir / "trajectories.jsonl").read_text().splitlines():
            line = line.strip()
            if line:
                out.append(Trajectory(**json.loads(line)))
        return out

    def labels(self) -> dict[str, bool]:
        return json.loads((self._dir / "labels.json").read_text())

    def check_drift(self, verdicts: list[Verdict], min_agreement: float = 0.85) -> DriftReport:
        labels = self.labels()
        by_id = {v.trajectory_id: v.passed for v in verdicts}
        agree = sum(1 for tid, expected in labels.items() if by_id.get(tid) == expected)
        rate = agree / len(labels) if labels else 1.0
        disagreements = [tid for tid, expected in labels.items() if by_id.get(tid) != expected]
        return DriftReport(agreement_rate=rate, disagreements=disagreements)
