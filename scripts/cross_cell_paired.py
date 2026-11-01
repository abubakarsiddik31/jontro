"""Within-goal paired comparison of register x script cells (no API calls).

Test goals are not confined to one cell: a goal's paraphrases are spread over
personas, so most test goals appear in two or three cells. That makes a paired
cross-cell comparison possible. For each cell and each goal the cell shares with
at least one other cell, we take the cell's decision-point rate minus the rate
of the same goal's decision points in all other cells, and average those
per-goal differences over the shared goals. The 95% interval resamples shared
goals (percentile bootstrap, 2,000 resamples, the scorer's fixed seed).

Writes outputs/stats/cross_cell_paired.json.

Usage:
    python scripts/cross_cell_paired.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from bangla_datasets.eval.multiturn_scoring import (
    SEED,
    _goal,
    score_step,
    step_predictions_from_rows,
)
from bangla_datasets.eval.multiturn_task import iter_decision_points
from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import load_split_parquet

TEST_PARQUET = Path("outputs/parquet/jontro_test.parquet")
EVAL = Path("outputs/eval")
OUT = Path("outputs/stats/cross_cell_paired.json")
MODELS = ["Qwen3.5-2B-zeroshot", "Qwen3.5-2B-SFT"]
METRICS = {"tool_selection": "names_match", "arg_exact": "args_exact"}
REPS = 2000


def _cell(traj: Trajectory) -> str:
    return f"{traj.persona.register}/{traj.persona.script}"


def _paired(per_goal: dict[str, dict[str, list[int]]], cell: str) -> dict:
    """Mean over shared goals of (rate in cell - rate in the goal's other cells)."""
    diffs = []
    shared: list[int] = []
    unshared: list[int] = []
    for cells in per_goal.values():
        if cell not in cells:
            continue
        inside = cells[cell]
        if len(cells) < 2:
            unshared += inside
            continue
        outside = [v for c, vs in cells.items() if c != cell for v in vs]
        diffs.append(100.0 * (sum(inside) / len(inside) - sum(outside) / len(outside)))
        shared += inside
    base = {
        "shared_goals": len(diffs),
        "points_in_cell": len(shared),
        "rate_shared_goals": 100.0 * sum(shared) / len(shared) if shared else None,
        "points_unshared_goals": len(unshared),
        "rate_unshared_goals": 100.0 * sum(unshared) / len(unshared) if unshared else None,
    }
    if len(diffs) < 3:
        return base
    rng = random.Random(SEED)
    draws = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(REPS)
    )
    return {
        **base,
        "mean_diff": sum(diffs) / len(diffs),
        "ci95": [draws[int(0.025 * REPS)], draws[int(0.975 * REPS)]],
    }


def main() -> None:
    trajs = {r["id"]: Trajectory.model_validate(r) for r in load_split_parquet(TEST_PARQUET)}
    goal_cells: dict[str, set[str]] = defaultdict(set)
    for t in trajs.values():
        goal_cells[_goal(t)].add(_cell(t))

    out: dict = {
        "test_goals": len(goal_cells),
        "goals_by_cells_spanned": {
            str(k): sum(1 for v in goal_cells.values() if len(v) == k)
            for k in sorted({len(v) for v in goal_cells.values()})
        },
        "method": "per-goal difference, cell minus the same goal's other cells; "
        "mean over shared goals; 95% percentile bootstrap over shared goals",
        "models": {},
    }
    for model in MODELS:
        rows = [
            json.loads(line)
            for line in (EVAL / f"multiturn_teacher_forced_{model}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        preds = {(p.trajectory_id, p.step): p for p in step_predictions_from_rows(rows)}
        per_metric: dict[str, dict[str, dict[str, list[int]]]] = {
            m: defaultdict(lambda: defaultdict(list)) for m in METRICS
        }
        for tid, traj in trajs.items():
            for dp in iter_decision_points(traj):
                p = preds.get((tid, dp.index))
                o = score_step(p.tool_calls if p else [], dp.gold_calls)
                for m, key in METRICS.items():
                    per_metric[m][_goal(traj)][_cell(traj)].append(int(o[key]))
        cells = sorted({c for v in goal_cells.values() for c in v})
        out["models"][model] = {
            m: {c: _paired(per_metric[m], c) for c in cells} for m in METRICS
        }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
