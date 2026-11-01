"""Score Modal teacher-forced benchmark outputs with the repo's own scorer, on the steps every arm ran.

Reads ``<src>/*.jsonl`` (from ``jontro_modal benchmark``), restricts scoring to the (trajectory, step) keys present in
every file, and prints tool selection, argument-exact and slot accuracy overall and by script, with goal-clustered CIs.

    python scripts/score_benchmark.py --src out_modal/benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bangla_datasets.eval.multiturn_scoring import score_rows
from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import load_split_parquet


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="out_modal/benchmark")
    ap.add_argument("--mlflow", action="store_true", help="log each arm as a run in the local mlflow.db (experiment jontro-benchmark)")
    args = ap.parse_args()
    trajs = {r["id"]: Trajectory.model_validate(r) for r in load_split_parquet("outputs/parquet/jontro_test.parquet")}
    files = {p.stem: [json.loads(line) for line in p.read_text().splitlines()] for p in sorted(Path(args.src).glob("*.jsonl"))}
    common = set.intersection(*({(r["example_id"], r["step"]) for r in rows} for rows in files.values()))
    out = {}
    print(f"steps scored (common to all {len(files)} arms): {len(common)}\n")
    print("| arm | tool-sel (95% CI) | arg-exact (95% CI) | slot-acc | no call | Bengali tool-sel | Banglish tool-sel |\n|---|---|---|---|---|---|---|")
    for arm, rows in files.items():
        s = score_rows("teacher_forced", rows, trajs, only=common)
        ov, bs = s.overall, s.by_script
        no_call = sum(1 for r in rows if (r["example_id"], r["step"]) in common and not r["tool_calls"]) / len(common)
        out[arm] = {"tool_selection": ov["tool_selection_rate"], "arg_exact": ov["arg_exact_rate"], "slot_accuracy": ov["slot_accuracy"],
                    "no_call_rate": no_call, "tool_selection_ci": ov["tool_selection_ci"], "arg_exact_ci": ov["arg_exact_ci"], "by_script": {k: v["tool_selection_rate"] for k, v in bs.items()}, "n": int(ov["n"])}
        def ci(c):
            return f" [{c[0]:.1f}, {c[1]:.1f}]" if c else ""
        print(f"| {arm} | {pct(ov['tool_selection_rate'])}{ci(ov['tool_selection_ci'])} | {pct(ov['arg_exact_rate'])}{ci(ov['arg_exact_ci'])} | {pct(ov['slot_accuracy'])} | {pct(no_call)} | "
              f"{pct(bs['bengali']['tool_selection_rate'])} | {pct(bs['banglish']['tool_selection_rate'])} |")
    Path(args.src, "benchmark_report.json").write_text(json.dumps(out, indent=1))
    if args.mlflow:
        log_mlflow(out, len(common))


def log_mlflow(out: dict, n_steps: int) -> None:
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("jontro-benchmark")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("jontro-benchmark").experiment_id
    for arm, r in out.items():
        if client.search_runs([exp], f"tags.mlflow.runName = 'bench-{arm}' and attributes.status = 'FINISHED'"):
            print(f"mlflow: bench-{arm} already logged, skipping")
            continue
        with mlflow.start_run(run_name=f"bench-{arm}"):
            mlflow.set_tags({"kind": "benchmark", "protocol": "teacher_forced", "smoke": "false"})
            mlflow.log_params({"arm": arm, "steps": n_steps, "decoding": "greedy, thinking off, max_new 300"})
            mlflow.log_metrics({"tool_selection": r["tool_selection"], "arg_exact": r["arg_exact"], "slot_accuracy": r["slot_accuracy"],
                                "no_call_rate": r["no_call_rate"], **{f"tool_selection_{k}": v for k, v in r["by_script"].items()}})
            if r.get("tool_selection_ci"):
                mlflow.log_metrics({"tool_selection_ci_lo_pct": r["tool_selection_ci"][0], "tool_selection_ci_hi_pct": r["tool_selection_ci"][1]})
            mlflow.log_artifact(f"out_modal/benchmark_full/{arm}.jsonl")


if __name__ == "__main__":
    main()
