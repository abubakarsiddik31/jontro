"""Attach deterministic validation flags to the released Jontro corpus.

The generation gates run at build time and reject candidates, but the released
records carried no per-item verdict, so a downstream user had nothing to filter
on and no way to see how each trajectory passed. This script recomputes the
deterministic checks offline — no model calls, no network — and writes them
under ``metadata.validation``, then regenerates every derived artifact so they
stay consistent with the canonical release.

Flags written per record:

  schema_pass            structural gate (schema_check)
  consistency_pass       simulator replay byte-match (consistency)
  heuristics_pass        script / length / repetition / register gate
  heuristic_reasons      the failing reasons, when heuristics_pass is False
  register_pronoun_flag  hard cross-register pronoun (unambiguous; gateable)
  register_imperative_flag  advisory cross-register imperative (bare stems
                         like কর/বল are shared between tui and tumi)
  register_verb_flag     advisory cross-register verb-suffix morphology
  sadhu_bhasha           archaic literary (Sadhu) register present anywhere
  reference_call         reference response makes >= 1 tool call
  defect_class           for no-call records, the rule-based class
  n_candidate_tools      k, the candidate-set size the model chooses among
  closure_style_present  whether metadata.closure_style is set
  judge_scored           always False (no LLM judge ran on this release)

Usage:  python scripts/add_release_flags.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bangla_datasets.schema import Role, Trajectory  # noqa: E402
from bangla_datasets.splits.io_parquet import write_split_parquet  # noqa: E402
from bangla_datasets.validation.consistency import check_consistency  # noqa: E402
from bangla_datasets.validation.heuristics import (  # noqa: E402
    contains_sadhu_bhasha,
    register_conformance,
    run_heuristics,
)
from bangla_datasets.validation.schema_check import check_schema  # noqa: E402
from bangla_datasets.writers.openai_format import write_openai_jsonl  # noqa: E402
from bangla_datasets.writers.sharegpt import write_sharegpt  # noqa: E402

RELEASE = ROOT / "outputs/release_multiturn"
CANONICAL = RELEASE / "jontro_multiturn_2000.jsonl"
OPENAI = RELEASE / "jontro_multiturn_2000_openai.jsonl"
SHAREGPT = RELEASE / "jontro_multiturn_2000_sharegpt.jsonl"
GOLDEN = RELEASE / "jontro_multiturn_100.jsonl"
REPORT = RELEASE / "generation_report.json"
PARQUET = ROOT / "outputs/parquet"
SPLITS = ROOT / "outputs/splits"

# Rule-based no-call classification (approximate; rules published here).
REFUSAL_RE = re.compile(
    r"(দুঃখিত|পার(ছি|বো|ব|ি) না|সম্ভব নয়|শুধু(মাত্র)?\s|only|sorry|cannot)", re.I
)
ASK_RE = re.compile(
    r"(\?|বলুন|জানান|জানাবেন|জানালে|জানতে চাই|কোনটি|কাকে|কত টাকা|কোন বিষয়"
    r"|কী ধরনের|কোথা থেকে|কোন স্টেশন|নাম কি|নম্বর কি|উল্লেখ করুন|প্রয়োজন হবে)"
)
ANNOUNCE_RE = re.compile(r"(করছি|খুঁজছি|দেখছি|করতে পারি|দিতে পারি|অনুসন্ধান)")


def reference_call(rec: dict) -> bool:
    return any(m.get("tool_calls") for m in rec["messages"])


def classify_no_gold(rec: dict) -> str:
    reply = " ".join(
        m.get("content") or "" for m in rec["messages"] if m.get("role") == "assistant"
    )
    if REFUSAL_RE.search(reply):
        return "refusal"
    if ASK_RE.search(reply):
        return "clarification_unanswered"
    if ANNOUNCE_RE.search(reply):
        return "announced_not_executed"
    return "unsourced_claim"


def sadhu_present(traj: Trajectory) -> bool:
    for m in traj.messages:
        if m.role in (Role.USER, Role.ASSISTANT) and m.content and contains_sadhu_bhasha(m.content):
            return True
    return False


def compute_flags(traj: Trajectory, rec: dict) -> dict:
    conf = register_conformance(traj)
    haves_call = reference_call(rec)
    return {
        "schema_pass": bool(check_schema(traj).passed),
        "consistency_pass": bool(check_consistency(traj).passed),
        "heuristics_pass": bool(run_heuristics(traj).passed),
        "heuristic_reasons": list(run_heuristics(traj).reasons or []),
        "register_pronoun_flag": bool(conf["pronoun_conflict"]),
        "register_imperative_flag": bool(conf["imperative_conflict"]),
        "register_verb_flag": bool(conf["verb_flag"]),
        "sadhu_bhasha": sadhu_present(traj),
        "reference_call": haves_call,
        "defect_class": None if haves_call else classify_no_gold(rec),
        "n_candidate_tools": len(rec.get("tools") or []),
        "closure_style_present": bool((rec.get("metadata") or {}).get("closure_style")),
        "judge_scored": False,
    }


def write_canonical(records: list[dict]) -> None:
    tmp = CANONICAL.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(CANONICAL)


def write_report(records: list[dict], flag_counts: Counter) -> None:
    domains = Counter(r.get("domain") for r in records)
    registers = Counter((r.get("persona") or {}).get("register") for r in records)
    scripts = Counter((r.get("persona") or {}).get("script") for r in records)
    closures = Counter((r.get("metadata") or {}).get("closure_style") for r in records)
    tools = [len((r.get("metadata") or {}).get("tools_used") or []) for r in records]
    msgs = [len(r.get("messages") or []) for r in records]
    total = len(records)
    report = {
        "total_trajectories": total,
        "domains": dict(sorted(domains.items())),
        "registers": dict(registers),
        "scripts": dict(scripts),
        "closure_styles": {str(k): v for k, v in closures.items()},
        "avg_tools_per_trajectory": sum(tools) / max(1, total),
        "avg_messages_per_trajectory": sum(msgs) / max(1, total),
        "validation_flags": dict(sorted(flag_counts.items())),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    records = [json.loads(line) for line in CANONICAL.open(encoding="utf-8") if line.strip()]
    golden_ids = [
        json.loads(line)["id"] for line in GOLDEN.open(encoding="utf-8") if line.strip()
    ]

    flag_counts: Counter = Counter()
    for rec in records:
        traj = Trajectory.model_validate(rec)
        flags = compute_flags(traj, rec)
        rec.setdefault("metadata", {})["validation"] = flags
        for key in (
            "schema_pass",
            "consistency_pass",
            "heuristics_pass",
            "register_pronoun_flag",
            "register_imperative_flag",
            "register_verb_flag",
            "sadhu_bhasha",
            "reference_call",
            "closure_style_present",
        ):
            if flags[key]:
                flag_counts[key] += 1
        if flags["defect_class"]:
            flag_counts[flags["defect_class"]] += 1
        if not flags["schema_pass"]:
            flag_counts["schema_fail"] += 1
        if not flags["consistency_pass"]:
            flag_counts["consistency_fail"] += 1
        if not flags["heuristics_pass"]:
            flag_counts["heuristics_fail"] += 1

    write_canonical(records)

    trajs = [Trajectory.model_validate(r) for r in records]
    write_openai_jsonl(trajs, OPENAI)
    write_sharegpt(trajs, SHAREGPT)

    by_id = {r["id"]: r for r in records}
    golden = [by_id[i] for i in golden_ids if i in by_id]
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for rec in golden:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_ids = {
        json.loads(line)["id"]
        for line in (SPLITS / "train_ids.jsonl").read_text().splitlines()
    }
    test_ids = {
        json.loads(line)["id"]
        for line in (SPLITS / "test_ids.jsonl").read_text().splitlines()
    }
    PARQUET.mkdir(parents=True, exist_ok=True)
    for split, ids in (("train", train_ids), ("test", test_ids)):
        write_split_parquet(
            trajs, ids, PARQUET / f"jontro_{split}.parquet", fmt="canonical"
        )
        write_split_parquet(
            trajs, ids, PARQUET / f"jontro_{split}_sharegpt.parquet", fmt="sharegpt"
        )

    write_report(records, flag_counts)

    print(f"flags attached to {len(records):,} records -> {CANONICAL}")
    print(
        "flag counts:",
        json.dumps(dict(sorted(flag_counts.items())), indent=1, ensure_ascii=False),
    )
    print(f"report -> {REPORT}")
    for p in sorted(PARQUET.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
