"""Carve a deterministic 90/10 train/test partition, stratified over
(register, script) — the two sociolinguistic axes that are the paper's
centerpiece. Stratification guarantees the test split represents every
axis bucket, so the baseline breakdowns are meaningful.

Register mapping (paper terminology):
    tui   = intimate (তুই/তোর)
    tumi  = familiar (তুমি/তোমাকে)
    apni  = formal/respectful (আপনি)
"""
from __future__ import annotations

import random
from collections import defaultdict

from bangla_datasets.schema import Trajectory

SEED = 20260716


def partition(
    trajectories: list[Trajectory],
    test_fraction: float = 0.10,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids), stratified over (register, script).

    Within each bucket we shuffle with a per-bucket seed derived from the
    global seed + bucket key, so adding a new bucket never reshuffles the
    others. Test takes the first ``test_fraction`` of each bucket; train
    takes the rest. Both lists are sorted by id for stable output.
    """
    buckets: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        buckets[(t.persona.register, t.persona.script)].append(t)

    train_ids: list[str] = []
    test_ids: list[str] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda t: t.id)
        bucket_seed = hash((seed, key)) & 0xFFFFFFFF
        rng = random.Random(bucket_seed)
        rng.shuffle(pool)
        k = max(1, round(len(pool) * test_fraction))
        test_ids.extend(t.id for t in pool[:k])
        train_ids.extend(t.id for t in pool[k:])

    train_ids.sort()
    test_ids.sort()
    return train_ids, test_ids


def _goal_of(t: Trajectory) -> str:
    """The task goal a trajectory instantiates (falls back to its id)."""
    return (t.metadata or {}).get("persona_goal") or t.id


def partition_by_goal(
    trajectories: list[Trajectory],
    test_fraction: float = 0.30,
    seed: int = SEED,
) -> tuple[list[str], list[str], set[str]]:
    """Return (train_ids, test_ids, test_goals), holding out whole task goals.

    Records share a small pool of underlying goals, each repeated across
    register, script, and persona paraphrases. An id-level split therefore leaks
    goals: a test record's goal almost always also appears in train, so the test
    split measures paraphrase recall, not transfer. This partitions at the
    *goal* level instead — every record of a held-out goal goes to test — so no
    test goal has a paraphrase in train.

    Goals are held out per domain so both sides keep every domain, and at least
    one goal per multi-goal domain is kept in train (a single-goal domain stays
    entirely in train). ``test_fraction`` is the fraction of *goals* held out,
    not records; since replication is uneven the record split will not match it
    exactly.

    Returns the train ids, the test ids, and the set of held-out goal strings
    (so callers can report and assert zero overlap).
    """
    goals_by_domain: dict[str, set[str]] = defaultdict(set)
    for t in trajectories:
        goals_by_domain[t.domain].add(_goal_of(t))

    rng = random.Random(seed)
    test_goals: set[str] = set()
    for domain in sorted(goals_by_domain):
        goals = sorted(goals_by_domain[domain])
        rng.shuffle(goals)
        k = round(len(goals) * test_fraction)
        # never empty the train side of a domain
        k = min(k, max(0, len(goals) - 1))
        test_goals.update(goals[:k])

    train_ids: list[str] = []
    test_ids: list[str] = []
    for t in trajectories:
        (test_ids if _goal_of(t) in test_goals else train_ids).append(t.id)

    train_ids.sort()
    test_ids.sort()
    return train_ids, test_ids, test_goals
