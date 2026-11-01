"""Precompute the exact teacher-forced test prompts (one per decision point) for the Modal benchmark job.

Uses the repo's own ``build_step_messages`` / ``build_step_tools`` so the container needs no repo code.
Arguments in the gold prefix become dicts, as chat templates require.

    python scripts/build_benchmark_prompts.py   # -> build/benchmark/teacher_forced_test.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.eval.multiturn_task import build_step_messages, build_step_tools, iter_decision_points
from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import load_split_parquet

OUT = Path("build/benchmark/teacher_forced_test.jsonl")


def dictify(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        for tc in m.get("tool_calls") or []:
            fn = tc["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"] or "{}")
        out.append(m)
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as f:
        for r in load_split_parquet("outputs/parquet/jontro_test.parquet"):
            t = Trajectory.model_validate(r)
            tools = build_step_tools(t)
            for dp in iter_decision_points(t):
                f.write(json.dumps({"example_id": t.id, "step": dp.index,
                                    "messages": dictify(build_step_messages(t, dp)), "tools": tools},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} decision points -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
