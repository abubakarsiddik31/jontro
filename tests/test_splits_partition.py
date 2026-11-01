"""Tests for the 90/10 stratified train/test partition."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bangla_datasets.schema import Message, Persona, Role, Trajectory
from bangla_datasets.splits.partition import partition, partition_by_goal

DATASET = Path("outputs/dataset/bangla_agentic.jsonl")
SEED = 20260716


def _load() -> list[Trajectory]:
    return [Trajectory.model_validate(json.loads(l)) for l in DATASET.read_text().splitlines()]


@pytest.fixture(scope="module")
def trajs():
    return _load()


@pytest.fixture(scope="module")
def splits(trajs):
    train_ids, test_ids = partition(trajs, test_fraction=0.10, seed=SEED)
    return train_ids, test_ids


def test_split_ratio_is_about_90_10(trajs, splits):
    train_ids, test_ids = splits
    n = len(trajs)
    assert abs(len(train_ids) - 0.90 * n) <= 0.02 * n
    assert abs(len(test_ids) - 0.10 * n) <= 0.02 * n


def test_no_overlap_between_train_and_test(splits):
    train_ids, test_ids = splits
    assert set(train_ids).isdisjoint(set(test_ids))


def test_union_covers_all_trajectories(trajs, splits):
    train_ids, test_ids = splits
    all_ids = {t.id for t in trajs}
    assert set(train_ids) | set(test_ids) == all_ids


def test_stratified_over_register_and_script(trajs, splits):
    """Test set must contain every (register, script) bucket proportionally."""
    by_bucket: dict[tuple[str, str], list[str]] = {}
    for t in trajs:
        key = (t.persona.register, t.persona.script)
        by_bucket.setdefault(key, []).append(t.id)
    _, test_ids = splits
    test_set = set(test_ids)
    for key, ids in by_bucket.items():
        in_test = sum(1 for i in ids if i in test_set)
        # Every bucket contributes roughly 10% to test (±10pp; small buckets
        # like banglish+tumi can't hit exactly 10%).
        frac = in_test / len(ids)
        assert 0.0 <= frac <= 0.25, f"bucket {key} test frac={frac:.2f}"


def test_partition_is_deterministic(trajs, splits):
    train_ids, _ = splits
    again_train, _ = partition(trajs, test_fraction=0.10, seed=SEED)
    assert train_ids == again_train


# --- Goal-disjoint partition -------------------------------------------------

def _mk(tid: str, domain: str, goal: str) -> Trajectory:
    return Trajectory(
        id=tid,
        domain=domain,
        persona=Persona(),
        tools=[],
        system_prompt="s",
        messages=[Message(role=Role.USER, content="x")],
        metadata={"persona_goal": goal},
    )


def _synthetic() -> list[Trajectory]:
    # d1: 4 goals x 10 records; d2: 2 goals x 5 records
    trajs = [_mk(f"a{i}", "d1", f"g{i % 4}") for i in range(40)]
    trajs += [_mk(f"b{i}", "d2", f"h{i % 2}") for i in range(10)]
    return trajs


def test_goal_disjoint_zero_goal_overlap() -> None:
    trajs = _synthetic()
    train, test, test_goals = partition_by_goal(trajs, test_fraction=0.5, seed=1)
    by_id = {t.id: t for t in trajs}
    train_goals = {by_id[i].metadata["persona_goal"] for i in train}
    got_test_goals = {by_id[i].metadata["persona_goal"] for i in test}
    assert not (train_goals & got_test_goals)
    assert got_test_goals == test_goals


def test_goal_disjoint_covers_all_and_keeps_each_domain() -> None:
    trajs = _synthetic()
    train, test, _ = partition_by_goal(trajs, test_fraction=0.5, seed=1)
    assert set(train) | set(test) == {t.id for t in trajs}
    assert set(train).isdisjoint(set(test))
    by_id = {t.id: t for t in trajs}
    # every domain must still appear on the train side
    train_domains = {by_id[i].domain for i in train}
    assert train_domains == {"d1", "d2"}


def test_goal_disjoint_deterministic() -> None:
    trajs = _synthetic()
    a = partition_by_goal(trajs, test_fraction=0.5, seed=1)
    b = partition_by_goal(trajs, test_fraction=0.5, seed=1)
    assert a == b


def test_shipped_split_is_goal_disjoint() -> None:
    """Regression: the released split must not leak goals into test."""
    trajs = [
        Trajectory.model_validate(json.loads(line))
        for line in DATASET.read_text().splitlines()
    ]
    by_id = {t.id: t for t in trajs}
    splits = Path("outputs/splits")
    train = {
        json.loads(line)["id"]
        for line in (splits / "train_ids.jsonl").read_text().splitlines()
    }
    test = {
        json.loads(line)["id"]
        for line in (splits / "test_ids.jsonl").read_text().splitlines()
    }
    train_goals = {by_id[i].metadata.get("persona_goal") for i in train}
    test_goals = {by_id[i].metadata.get("persona_goal") for i in test}
    assert not (train_goals & test_goals)

