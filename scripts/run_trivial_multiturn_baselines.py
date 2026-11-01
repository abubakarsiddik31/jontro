"""Non-learned baselines for the multi-turn test split (no API calls).

These are reference floors, not model baselines. Each emits one tool call per
decision point in the teacher-forced prediction format, so the standard scorer
(``scripts/score_multiturn.py``) consumes them unchanged.

- majority-global : always call the single most frequent tool in the train split.
- majority-domain : call the most frequent tool for the trajectory's domain.
- first-offered   : call the first tool in the trajectory's offered set.
- random-offered  : seeded uniform draw from the trajectory's offered set.

Reported alongside the ORACLE ceiling (gold calls replayed, 100%). They show the
task is not solvable by prior or position alone, and they exercise the harness
without spending API budget.

Usage:
    python scripts/run_trivial_multiturn_baselines.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import load_split_parquet

TRAIN = Path("outputs/parquet/jontro_train.parquet")
TEST = Path("outputs/parquet/jontro_test.parquet")
OUT = Path("outputs/eval")
SEED = 20260725

Pick = Callable[[Trajectory, list[str], random.Random], str]


def _trajs(path: Path) -> list[Trajectory]:
    return [Trajectory.model_validate(r) for r in load_split_parquet(path)]


def _global_majority(trajs: list[Trajectory]) -> str:
    counts: Counter = Counter()
    for t in trajs:
        for name in (t.metadata.get("tools_used") or []):
            counts[name] += 1
    return counts.most_common(1)[0][0]


def _domain_majority(trajs: list[Trajectory]) -> dict[str, str]:
    per: dict[str, Counter] = {}
    for t in trajs:
        per.setdefault(t.domain, Counter())
        for name in (t.metadata.get("tools_used") or []):
            per[t.domain][name] += 1
    return {d: c.most_common(1)[0][0] for d, c in per.items() if c}


def _emit(name: str, test: list[Trajectory], pick: Pick) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"multiturn_teacher_forced_{name}.jsonl"
    rng = random.Random(SEED)
    with path.open("w", encoding="utf-8") as f:
        for t in test:
            offered = [tool.name for tool in t.tools]
            step = 0
            for m in t.messages:
                if m.role.value != "assistant" or not m.tool_calls:
                    continue
                call_name = pick(t, offered, rng)
                row = {
                    "example_id": t.id,
                    "step": step,
                    "protocol": "teacher_forced",
                    "model": name,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "name": call_name,
                            "arguments": {},
                            "thought_signature_b64": None,
                        }
                    ],
                    "raw_text": "",
                    "error": None,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                step += 1
    return path


def main() -> None:
    train, test = _trajs(TRAIN), _trajs(TEST)
    g = _global_majority(train)
    dom = _domain_majority(train)
    print(f"train={len(train)} test={len(test)} global_majority={g}")

    paths = [
        _emit("majority-global", test, lambda t, o, r: g),
        _emit("majority-domain", test, lambda t, o, r: dom.get(t.domain, g)),
        _emit("first-offered", test, lambda t, o, r: o[0] if o else g),
        _emit("random-offered", test, lambda t, o, r: r.choice(o) if o else g),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
