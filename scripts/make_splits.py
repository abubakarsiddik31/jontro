"""Carve the goal-disjoint train/test partition and write id manifests.

The shipped split is **goal-disjoint** (not id-level): whole task goals are held
out so no test record's goal has a paraphrase in train. See
``bangla_datasets.splits.partition.partition_by_goal`` for the rationale.

Usage:
    python scripts/make_splits.py
"""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits import partition_by_goal

DATASET = Path("outputs/dataset/bangla_agentic.jsonl")
OUT = Path("outputs/splits")
TEST_GOAL_FRACTION = 0.30
SEED = 20260725


def _goal(t: Trajectory) -> str:
    return (t.metadata or {}).get("persona_goal") or t.id


def main() -> None:
    trajs = [
        Trajectory.model_validate(json.loads(line))
        for line in DATASET.read_text().splitlines()
    ]
    train_ids, test_ids, test_goals = partition_by_goal(
        trajs, test_fraction=TEST_GOAL_FRACTION, seed=SEED
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_ids.jsonl").write_text(
        "\n".join(json.dumps({"id": i}) for i in train_ids) + "\n"
    )
    (OUT / "test_ids.jsonl").write_text(
        "\n".join(json.dumps({"id": i}) for i in test_ids) + "\n"
    )
    # Manifest of the held-out goals, so the split's transfer property is auditable.
    (OUT / "test_goals.jsonl").write_text(
        "\n".join(json.dumps({"goal": g}, ensure_ascii=False) for g in sorted(test_goals))
        + "\n"
    )

    # Invariant: zero goal overlap between the two sides.
    by_id = {t.id: t for t in trajs}
    train_goals = {_goal(by_id[i]) for i in train_ids}
    test_set = {_goal(by_id[i]) for i in test_ids}
    assert not (train_goals & test_set), "goal leak between train and test"

    n = len(trajs)
    print(
        f"train: {len(train_ids)}  test: {len(test_ids)}  "
        f"(test {100 * len(test_ids) / n:.1f}% of records)  -> {OUT}"
    )
    print(
        f"train goals: {len(train_goals)}  test goals: {len(test_set)}  "
        f"overlap: {len(train_goals & test_set)}"
    )


if __name__ == "__main__":
    main()
