"""CLI: generate | validate | publish."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before any client reads keys.
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")

from rich.console import Console  # noqa: E402
from rich.progress import track  # noqa: E402

from bangla_datasets.gemini.client import GeminiClient  # noqa: E402
from bangla_datasets.generation.coverage import CoverageMatrix  # noqa: E402
from bangla_datasets.generation.orchestrator import SelfPlayOrchestrator  # noqa: E402
from bangla_datasets.generation.seeds import load_recipe, sample_seed  # noqa: E402
from bangla_datasets.schema import Trajectory  # noqa: E402
from bangla_datasets.utils.checkpoint import Checkpoint  # noqa: E402
from bangla_datasets.utils.logging import configure_logging  # noqa: E402
from bangla_datasets.utils.seeding import SeededRNG  # noqa: E402
from bangla_datasets.validation.golden import GoldenSet  # noqa: E402
from bangla_datasets.validation.pipeline import ValidationPipeline  # noqa: E402
from bangla_datasets.writers.hf import publish_to_hf  # noqa: E402
from bangla_datasets.writers.jsonl import write_jsonl, write_verdicts  # noqa: E402
from bangla_datasets.writers.sharegpt import write_sharegpt  # noqa: E402

console = Console()
_log = logging.getLogger("bangla_datasets.cli")


def _cmd_generate(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    client = GeminiClient(model=args.model, max_input_tokens_per_run=args.cost_cap)
    orch = SelfPlayOrchestrator(client=client, max_turns=recipe.max_turns)
    cp = Checkpoint(run_dir)
    coverage = CoverageMatrix()
    rng = SeededRNG(seed=args.master_seed)

    target = args.count or recipe.target_count
    start = args.start
    console.print(
        f"[bold]Generating {target} trajectories "
        f"(indices {start}–{start + target - 1})…[/bold]"
    )

    # Incremental write (FIX 4): each trajectory is appended as it completes, so
    # a mid-run crash never loses work. On a fresh run (empty checkpoint) we
    # truncate any stale file; on a resume we append.
    out_path = run_dir / "trajectories.jsonl"
    # Reconcile checkpoint against the file first: a killed process can leave
    # IDs marked done whose line never made it to disk. This prunes them so they
    # regenerate rather than being silently lost.
    cp.reconcile(out_path)
    file_mode = "a" if len(cp._done) > 0 else "w"  # noqa: SLF001 - resume detection
    failures = 0
    written = 0
    with out_path.open(file_mode, encoding="utf-8") as f:
        for offset in track(range(target), description="Generating"):
            i = start + offset
            traj_id = f"bd-agentic-{i:06d}"
            if cp.is_done(traj_id):
                continue
            plan = sample_seed(i, rng, recipe)
            coverage.record(plan)
            try:
                traj = orch.generate(plan)
            except Exception as e:  # noqa: BLE001 - per-trajectory isolation
                failures += 1
                console.print(f"[yellow]Failed trajectory {i}: {e}[/yellow]")
                _log.warning("failed trajectory %d: %s", i, e)
                continue  # not marked done -> regenerates on resume
            # traj.id is f"bd-agentic-{plan.seed:06d}" == f"bd-agentic-{i:06d}"
            # since sample_seed sets seed=idx (see test_sample_seed_aligns_with_loop_index).
            f.write(traj.model_dump_json(ensure_ascii=False) + "\n")
            f.flush()
            cp.mark_done(traj.id)
            written += 1
            _log.info("generated %s", traj.id)

    console.print(
        f"[green]Wrote {written} trajectories to {out_path}"
        + (f" ({failures} failed)[/green]" if failures else "[/green]")
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    client = GeminiClient(model=args.model)
    pipeline = ValidationPipeline(client=client, pass_threshold=args.threshold)
    # Load trajectories.
    trajectories = [
        Trajectory.model_validate_json(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
        if line.strip()
    ]
    # Golden-set drift check first — a hedge against silent judge degradation.
    golden = GoldenSet()
    golden_verdicts = [pipeline.validate(t) for t in golden.load()]
    drift = golden.check_drift(golden_verdicts)
    console.print(
        f"[bold]Golden agreement: {drift.agreement_rate:.0%}[/bold]  "
        f"disagreements: {drift.disagreements}"
    )
    if drift.agreement_rate < 0.85:
        console.print("[red]Judge drift detected — pausing. Review golden set + rubric.[/red]")
        return 2

    verdicts = []
    passed = 0
    for t in track(trajectories, description="Validating"):
        v = pipeline.validate(t)
        verdicts.append(v)
        if v.passed:
            passed += 1
    write_verdicts(verdicts, run_dir / "verdicts.jsonl")
    rate = passed / max(1, len(verdicts))
    console.print(f"[green]Passed: {passed}/{len(verdicts)} ({rate:.0%})[/green]")
    if rate < args.min_pass_rate:
        console.print(
            f"[red]Pass rate below threshold {args.min_pass_rate:.0%} "
            "— recipe needs review.[/red]"
        )
        return 3
    # Write only-passed trajectories.
    passed_ids = {v.trajectory_id for v in verdicts if v.passed}
    passed_trajs = [t for t in trajectories if t.id in passed_ids]
    write_jsonl(passed_trajs, run_dir / "validated.jsonl")
    write_sharegpt(passed_trajs, run_dir / "validated.sharegpt.jsonl")
    console.print(f"[green]Wrote {len(passed_trajs)} validated trajectories.[/green]")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    trajectories = [
        Trajectory.model_validate_json(line)
        for line in (run_dir / "validated.jsonl").read_text().splitlines()
        if line.strip()
    ]
    url = publish_to_hf(trajectories, repo_id=args.repo_id)
    console.print(f"[green]Published to {url}[/green]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bangla-datasets")
    default_model = os.getenv("BANGLA_DATASETS_MODEL", "gemini-2.5-flash")
    parser.add_argument("--model", default=default_model)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate trajectories.")
    g.add_argument("--recipe", default="recipes/agentic_v1.yaml")
    g.add_argument("--run-dir", default="runs/default")
    g.add_argument("--count", type=int, default=None)
    g.add_argument("--start", type=int, default=0, help="Starting index (for parallel batches).")
    g.add_argument("--master-seed", type=int, default=42)
    g.add_argument(
        "--cost-cap",
        type=int,
        default=50_000_000,
        help="Approx input-token budget per run (estimate). Bounds spend; "
        "uses real usage_metadata when available. Set 0 to disable.",
    )
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("validate", help="Validate + filter trajectories.")
    v.add_argument("--run-dir", default="runs/default")
    v.add_argument("--threshold", type=int, default=4)
    v.add_argument("--min-pass-rate", type=float, default=0.70)
    v.set_defaults(func=_cmd_validate)

    p = sub.add_parser("publish", help="Publish validated set to Hugging Face.")
    p.add_argument("--run-dir", default="runs/default")
    p.add_argument("--repo-id", required=True)
    p.set_defaults(func=_cmd_publish)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
