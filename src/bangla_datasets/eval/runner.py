"""Run one model over the test split, streaming predictions to JSONL.

Resumable: if the output file already contains a row for an example id, that
id is skipped on re-run. This lets long runs recover from interruptions and
makes it cheap to tweak model ids and re-run.

Rate limiting: sleeps ``min_delay`` seconds between requests to stay within
free-tier RPM limits (Groq free tier is ~30 rpm). Errors are still retried
with backoff inside the client; this delay keeps us *under* the limit rather
than recovering from it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bangla_datasets.schema import Trajectory

_log = logging.getLogger("bangla_datasets.eval")

# Groq free tier: ~30 requests/min => ~2s floor between requests.
DEFAULT_MIN_DELAY = 3.0


class _ClientProto(Protocol):
    model: str

    def predict(self, traj: Trajectory):  # noqa: ANN001
        ...


@dataclass
class RunnerReport:
    model: str
    n: int
    n_errors: int
    out_path: Path


def _already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_model(
    client: _ClientProto,
    trajectories: list[Trajectory],
    out_path: Path,
    min_delay: float = DEFAULT_MIN_DELAY,
) -> RunnerReport:
    """Predict for each trajectory not already in out_path; append results."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(out_path)
    n_errors = 0
    n = 0
    with out_path.open("a") as f:
        for t in trajectories:
            if t.id in done:
                n += 1
                continue
            if n > 0 and min_delay > 0:
                time.sleep(min_delay)
            pred = client.predict(t)
            tc = pred.tool_call
            row = {
                "example_id": t.id,
                "model": client.model,
                "tool_call": tc.model_dump(mode="json") if tc else None,
                "raw_text": pred.raw_text,
                "error": pred.error,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if pred.error:
                n_errors += 1
            if n % 10 == 0:
                _log.info(
                    "%s: %d/%d (errors=%d)", client.model, n, len(trajectories), n_errors
                )
    return RunnerReport(model=client.model, n=n, n_errors=n_errors, out_path=out_path)
