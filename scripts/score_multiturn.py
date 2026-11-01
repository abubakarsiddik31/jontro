"""Score multi-turn predictions and render the report artifacts (no API calls).

Reads ``outputs/eval/multiturn_<protocol>_<MODEL>.jsonl`` produced by
``run_multiturn_baselines.py``, joins each against the goal-disjoint test split,
and reports per-step tool selection, argument-value correctness, slot accuracy,
per-turn-index degradation, task completion, plus register/script/domain
breakdowns — all with goal-clustered 95% CIs.

Usage:
    python scripts/score_multiturn.py                      # all files found
    python scripts/score_multiturn.py --protocol interactive
    python scripts/score_multiturn.py --models ORACLE-gold,GPT-OSS-20B
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

from bangla_datasets.eval.multiturn_scoring import MultiTurnScore, score_rows  # noqa: E402
from bangla_datasets.schema import Trajectory  # noqa: E402
from bangla_datasets.splits.io_parquet import load_split_parquet  # noqa: E402

TEST_PARQUET = Path("outputs/parquet/jontro_test.parquet")
OUT = Path("outputs/eval")


def load_test_trajectories() -> dict[str, Trajectory]:
    return {r["id"]: Trajectory.model_validate(r) for r in load_split_parquet(TEST_PARQUET)}


def discover(protocol: str) -> dict[str, Path]:
    files = sorted(OUT.glob(f"multiturn_{protocol}_*.jsonl"))
    return {p.stem.replace(f"multiturn_{protocol}_", ""): p for p in files}


def _ci(ci: tuple[float, float] | None) -> str:
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "n/a"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def print_report(name: str, s: MultiTurnScore) -> None:
    ov = s.overall
    print(f"\n=== {name}  ({s.status}, {s.n_trajectories} trajs, {s.n_goals} goals) ===")
    print(f"  steps scored:        {int(ov['n'])}")
    print(
        f"  tool-selection:      {_pct(ov['tool_selection_rate'])}%  "
        f"CI {_ci(ov['tool_selection_ci'])}"
    )
    print(f"  arg-exact (step):    {_pct(ov['arg_exact_rate'])}%  CI {_ci(ov['arg_exact_ci'])}")
    print(f"  slot-accuracy:       {_pct(ov['slot_accuracy'])}%  CI {_ci(ov['slot_accuracy_ci'])}")
    tc = s.task_completion
    print(f"  task completion:     {_pct(tc['completion_rate'])}%  CI {_ci(tc['completion_ci'])}")
    print(f"  errors: {s.error_taxonomy}")
    print("  by turn index (tool-selection%):")
    for k, b in s.by_turn_index.items():
        print(f"    turn {k:>2}: n={int(b['n']):4d}  tm={_pct(b['tool_selection_rate'])}%  "
              f"exact={_pct(b['arg_exact_rate'])}%")
    print("  by register:")
    for k, b in s.by_register.items():
        print(f"    {k:8s} n={int(b['n']):4d}  tm={_pct(b['tool_selection_rate'])}%  "
              f"exact={_pct(b['arg_exact_rate'])}%")
    print("  by script:")
    for k, b in s.by_script.items():
        print(f"    {k:8s} n={int(b['n']):4d}  tm={_pct(b['tool_selection_rate'])}%  "
              f"exact={_pct(b['arg_exact_rate'])}%")


def write_markdown(scores: dict[str, MultiTurnScore], out_dir: Path) -> Path:
    md = out_dir / "multiturn_results.md"
    header = (
        "| Model | Steps | Tool-sel (%) | Arg-exact (%) | Slot-acc (%) | Completion (%) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, s in scores.items():
        ov, tc = s.overall, s.task_completion
        rows.append(
            f"| {name} | {int(ov['n'])} | {_pct(ov['tool_selection_rate'])} | "
            f"{_pct(ov['arg_exact_rate'])} | {_pct(ov['slot_accuracy'])} | "
            f"{_pct(tc['completion_rate'])} |"
        )
    body = header + "\n".join(rows) + "\n"
    # Per-turn-index table (shows the chained-degradation story).
    turns = sorted({int(k) for s in scores.values() for k in s.by_turn_index})
    body += "\n\n### Tool-selection by turn index (%)\n\n"
    body += "| Model | " + " | ".join(f"turn {t}" for t in turns) + " |\n"
    body += "|---" * (len(turns) + 1) + "|\n"
    for name, s in scores.items():
        cells = [
            _pct(s.by_turn_index[str(t)]["tool_selection_rate"])
            if str(t) in s.by_turn_index
            else "-"
            for t in turns
        ]
        body += f"| {name} | " + " | ".join(cells) + " |\n"
    md.write_text(body, encoding="utf-8")
    return md


def write_by_turn_figure(scores: dict[str, MultiTurnScore], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, s in scores.items():
        turns = sorted(int(k) for k in s.by_turn_index)
        vals = [s.by_turn_index[str(t)]["tool_selection_rate"] for t in turns]
        ax.plot([t + 1 for t in turns], vals, marker="o", label=name)
    ax.set_xlabel("decision point in conversation")
    ax.set_ylabel("tool-selection rate")
    ax.set_title("Multi-turn tool selection across chained steps")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "multiturn_by_turn.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--protocol", choices=["teacher_forced", "interactive"], default="teacher_forced"
    )
    ap.add_argument(
        "--models", default=None, help="comma-separated model names (default: all found)"
    )
    args = ap.parse_args()

    trajs = load_test_trajectories()
    files = discover(args.protocol)
    names = [m.strip() for m in args.models.split(",")] if args.models else list(files)
    if not names:
        print(f"No multiturn_{args.protocol}_*.jsonl files in {OUT}/.")
        return

    scores: dict[str, MultiTurnScore] = {}
    for name in names:
        path = files.get(name)
        if path is None:
            print(f"  (no file for {name}, skipping)")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        scores[name] = score_rows(args.protocol, rows, trajs)
        print_report(name, scores[name])

    OUT.mkdir(parents=True, exist_ok=True)
    md = write_markdown(scores, OUT)
    fig = write_by_turn_figure(scores, OUT) if scores else None
    # Machine-readable scores so the paper's tables are generated, never typed.
    (OUT / "multiturn_scores.json").write_text(
        json.dumps({k: asdict(v) for k, v in scores.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nReport: {md}")
    print(f"Scores: {OUT / 'multiturn_scores.json'}")
    if fig:
        print(f"Figure: {fig}")


if __name__ == "__main__":
    sys.exit(main())
