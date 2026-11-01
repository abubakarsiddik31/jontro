"""Run baseline models on the goal-disjoint multi-turn test split.

Two protocols (see ``bangla_datasets.eval.multiturn_runner``):

    --protocol teacher_forced   one prediction per decision point, gold prefix
    --protocol interactive      model drives every step, simulated tool results

Add ``--oracle`` for an offline, no-API self-check that echoes the gold calls
(expected: 100% on every metric). That is the harness/scorer smoke test.

Usage:
    # offline upper bound, no API key needed
    python scripts/run_multiturn_baselines.py --protocol teacher_forced --oracle

    # real run against an OpenAI-compatible endpoint
    EVAL_API_KEY=... EVAL_BASE_URL=https://api.groq.com/openai/v1 \
        python scripts/run_multiturn_baselines.py --protocol teacher_forced

Resumable: finished (example_id, step) units are skipped, so re-running only
fills gaps. ``--limit`` caps how many trajectories with remaining work this
invocation processes (rate-limited free tiers).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from bangla_datasets.eval.client import EvalClient  # noqa: E402
from bangla_datasets.eval.multiturn_runner import (  # noqa: E402
    run_interactive,
    run_teacher_forced,
)
from bangla_datasets.eval.multiturn_task import iter_decision_points  # noqa: E402
from bangla_datasets.schema import Trajectory  # noqa: E402
from bangla_datasets.splits.io_parquet import load_split_parquet  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)

TEST_PARQUET = Path("outputs/parquet/jontro_test.parquet")
OUT = Path("outputs/eval")

# Model configs; override with EVAL_MODELS="Name=id,Name=id".
DEFAULT_MODELS = [
    {"name": "GPT-OSS-120B", "model": "openai/gpt-oss-120b"},
    {"name": "GPT-OSS-20B", "model": "openai/gpt-oss-20b"},
]


def _models() -> list[dict]:
    spec = os.environ.get("EVAL_MODELS")
    if not spec:
        return DEFAULT_MODELS
    out = []
    for part in spec.split(","):
        name, _, mid = part.partition("=")
        out.append({"name": name.strip(), "model": (mid or name).strip()})
    return out


def load_test_trajectories() -> list[Trajectory]:
    return [Trajectory.model_validate(r) for r in load_split_parquet(TEST_PARQUET)]


def remaining_work(
    trajectories: list[Trajectory], out_path: Path, protocol: str, limit: int
) -> list[Trajectory]:
    """Trajectories with any unfinished unit, capped at *limit* (0 = no cap)."""
    if out_path.exists():
        rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    else:
        rows = []
    if protocol == "interactive":
        done = {r["example_id"] for r in rows}
        has_work = lambda t: t.id not in done  # noqa: E731
    else:
        done = {f"{r['example_id']}#{r['step']}" for r in rows}
        has_work = lambda t: not all(  # noqa: E731
            f"{t.id}#{dp.index}" in done for dp in iter_decision_points(t)
        )
    out = [t for t in trajectories if has_work(t)]
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protocol", choices=["teacher_forced", "interactive"], default="teacher_forced"
    )
    ap.add_argument("--oracle", action="store_true", help="echo gold calls; no API calls")
    ap.add_argument("--limit", type=int, default=0, help="cap trajectories with work (0 = all)")
    ap.add_argument("--delay", type=float, default=None, help="seconds between requests")
    args = ap.parse_args()

    trajectories = load_test_trajectories()
    print(f"Loaded {len(trajectories)} test trajectories ({args.protocol})")

    configs = [{"name": "ORACLE-gold", "model": "oracle"}] if args.oracle else _models()
    delay = args.delay if args.delay is not None else float(os.environ.get("EVAL_DELAY", "3.0"))
    runner = run_teacher_forced if args.protocol == "teacher_forced" else run_interactive

    for cfg in configs:
        out_path = OUT / f"multiturn_{args.protocol}_{cfg['name']}.jsonl"
        todo = remaining_work(trajectories, out_path, args.protocol, args.limit)
        if not todo:
            print(f"{cfg['name']}: nothing to do (already complete).")
            continue
        client = None if args.oracle else EvalClient(model=cfg["model"])
        report = runner(client, todo, out_path, min_delay=delay, oracle=args.oracle)
        print(
            f"{cfg['name']}: {report.n} units this run -> {report.out_path} "
            f"(errors={report.n_errors})"
        )


if __name__ == "__main__":
    main()
