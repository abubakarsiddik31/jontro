"""Multi-turn scoring: per-step tool selection, argument-value correctness,
task completion, with goal-clustered confidence intervals.

Two things this fixes relative to the old single-turn scorer (per the LRE
review): argument correctness compares **values** against gold (Bengali and
ASCII digits unified) instead of only checking JSON-schema validity, and every
metric is computed **per decision point** over the whole chain, not just the
first call.

Reporting unit: because the 2,000 records instantiate only 153 goals, records
sharing a goal are paraphrases of one task and move together. Confidence
intervals therefore resample **goals**, not records (``cluster_bootstrap``).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from bangla_datasets.eval.multiturn_task import iter_decision_points
from bangla_datasets.schema import ToolCall, Trajectory

BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
SEED = 20260725


# --------------------------------------------------------------------------
# argument-value comparison
# --------------------------------------------------------------------------
def norm_value(v: object) -> str:
    """Normalize an argument value for value-level comparison.

    Bengali and ASCII digits are unified (the corpus writes amounts, phone
    numbers and dates in both), case and separating punctuation are ignored.
    Values are otherwise compared literally — no synonym matching — so this is
    a strict measure.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        s = str(v)
    elif isinstance(v, (list, dict)):
        s = json.dumps(v, sort_keys=True, ensure_ascii=False)
    else:
        s = str(v)
    s = s.translate(BENGALI_DIGITS).strip().casefold()
    s = re.sub(r"[\s,]+", " ", s)
    return s.strip(" .।-")


def _call_key(tc: ToolCall) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Canonical, order-insensitive key for a call (name + normalized args).

    ``null``-valued entries are dropped: the corpus writes only the parameters a
    call actually uses, and a model that echoes the merged parameter set padded
    with nulls is not penalized for the padding.
    """
    args = tuple(
        sorted((k, norm_value(v)) for k, v in (tc.arguments or {}).items() if v is not None)
    )
    return (tc.name, args)


def score_step(pred: list[ToolCall], gold: list[ToolCall]) -> dict:
    """Score one decision point's predicted call set against the gold set.

    Parallel calls are treated as an unordered set: names are compared as a
    multiset and calls are paired by name for slot accounting. Slot totals count
    only non-null gold arguments.
    """
    pred_names = Counter(c.name for c in pred)
    gold_names = Counter(c.name for c in gold)
    name_hits = sum((pred_names & gold_names).values())
    first_match = bool(pred) and bool(gold) and pred[0].name == gold[0].name

    if not pred:
        return {
            "names_match": False,
            "args_exact": False,
            "first_match": False,
            "name_hits": 0,
            "name_gold": len(gold),
            "slots_ok": 0,
            "slots_tot": sum(
                1 for c in gold for v in (c.arguments or {}).values() if v is not None
            ),
            "k": len(gold),
            "kind": "no_call",
        }

    names_match = pred_names == gold_names
    args_exact = sorted(_call_key(c) for c in pred) == sorted(_call_key(c) for c in gold)

    slots_ok = slots_tot = 0
    if names_match:
        remaining: dict[str, list[ToolCall]] = defaultdict(list)
        for c in pred:
            remaining[c.name].append(c)
        for g in gold:
            gargs = {k: v for k, v in (g.arguments or {}).items() if v is not None}
            slots_tot += len(gargs)
            pool = remaining[g.name]
            p = pool.pop(0) if pool else None
            if p is None:
                continue
            pargs = p.arguments or {}
            slots_ok += sum(
                1 for k, v in gargs.items() if k in pargs and norm_value(pargs[k]) == norm_value(v)
            )

    if args_exact:
        kind = "correct"
    elif not names_match:
        kind = "wrong_tool"
    else:
        kind = "wrong_arg_values"

    return {
        "names_match": names_match,
        "args_exact": args_exact,
        "first_match": first_match,
        "name_hits": name_hits,
        "name_gold": len(gold),
        "slots_ok": slots_ok,
        "slots_tot": slots_tot,
        "k": len(gold),
        "kind": kind,
    }


# --------------------------------------------------------------------------
# goal-clustered bootstrap
# --------------------------------------------------------------------------
def cluster_bootstrap(
    items: list[tuple[str, int, int]], reps: int = 2000, seed: int = SEED
) -> tuple[float, float] | None:
    """95% CI (in percent) for a rate, resampling goals with replacement.

    ``items`` is a list of ``(goal, numerator, denominator)`` per unit. Units
    sharing a goal move together, which is the point: paraphrases of one task
    are not independent evidence. Returns ``None`` when there are < 3 goals.
    """
    by_goal: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for g, n, d in items:
        by_goal[g].append((n, d))
    goals = list(by_goal)
    if len(goals) < 3:
        return None
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(reps):
        num = den = 0
        for _ in goals:
            for n, d in by_goal[rng.choice(goals)]:
                num += n
                den += d
        if den:
            draws.append(100.0 * num / den)
    if not draws:
        return None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


# --------------------------------------------------------------------------
# predictions + aggregate result
# --------------------------------------------------------------------------
@dataclass
class StepPrediction:
    """One predicted decision point (teacher-forced protocol)."""

    trajectory_id: str
    step: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None


@dataclass
class MultiTurnScore:
    overall: dict
    by_turn_index: dict[str, dict]
    by_register: dict[str, dict]
    by_script: dict[str, dict]
    by_domain: dict[str, dict]
    by_k: dict[str, dict]
    error_taxonomy: dict[str, int]
    task_completion: dict
    n_trajectories: int
    n_goals: int
    status: str = "ok"  # "ok" | "partial" (some decision points missing)


def _goal(traj: Trajectory) -> str:
    return (traj.metadata or {}).get("persona_goal") or traj.id


def _bucket() -> dict:
    return {
        "n": 0,
        "names_match": 0,
        "args_exact": 0,
        "first_match": 0,
        "name_hits": 0,
        "name_gold": 0,
        "slots_ok": 0,
        "slots_tot": 0,
    }


def _bucket_dict(b: dict) -> dict:
    return {
        "n": float(b["n"]),
        "tool_selection_rate": _rate(b["names_match"], b["n"]),
        "first_call_accuracy": _rate(b["first_match"], b["n"]),
        "name_recall": _rate(b["name_hits"], b["name_gold"]),
        "arg_exact_rate": _rate(b["args_exact"], b["n"]),
        "slot_accuracy": _rate(b["slots_ok"], b["slots_tot"]),
    }


def score_teacher_forced(
    predictions: list[StepPrediction],
    trajectories: dict[str, Trajectory],
    only: set[tuple[str, int]] | None = None,
) -> MultiTurnScore:
    """Aggregate teacher-forced step predictions over the whole chain.

    ``only`` restricts scoring to those (trajectory_id, step) keys, for sub-sampled screening runs.

    Every gold decision point in ``trajectories`` is scored; a point with no
    prediction is counted as ``no_call`` (so a partial run yields honest,
    downward-biased numbers plus ``status == 'partial'``).
    """
    pred_by_key: dict[tuple[str, int], StepPrediction] = {
        (p.trajectory_id, p.step): p for p in predictions
    }

    overall = _bucket()
    by_turn: dict[int, dict] = defaultdict(_bucket)
    by_register: dict[str, dict] = defaultdict(_bucket)
    by_script: dict[str, dict] = defaultdict(_bucket)
    by_domain: dict[str, dict] = defaultdict(_bucket)
    by_k: dict[int, dict] = defaultdict(_bucket)
    errors: Counter[str] = Counter()

    sel_items: list[tuple[str, int, int]] = []
    first_items: list[tuple[str, int, int]] = []
    recall_items: list[tuple[str, int, int]] = []
    exact_items: list[tuple[str, int, int]] = []
    slot_items: list[tuple[str, int, int]] = []
    completion_items: list[tuple[str, int, int]] = []

    n_units = 0
    missing = 0
    for tid, traj in trajectories.items():
        dps = iter_decision_points(traj)
        if not dps:
            continue
        goal = _goal(traj)
        traj_all_exact = True
        for dp in dps:
            if only is not None and (tid, dp.index) not in only:
                continue
            n_units += 1
            pred = pred_by_key.get((tid, dp.index))
            if pred is None:
                missing += 1
                calls: list[ToolCall] = []
            else:
                calls = pred.tool_calls
            o = score_step(calls, dp.gold_calls)

            for b in (
                overall,
                by_turn[dp.index],
                by_register[traj.persona.register],
                by_script[traj.persona.script],
                by_domain[traj.domain],
                by_k[o["k"]],
            ):
                b["n"] += 1
                b["names_match"] += int(o["names_match"])
                b["args_exact"] += int(o["args_exact"])
                b["first_match"] += int(o["first_match"])
                b["name_hits"] += o["name_hits"]
                b["name_gold"] += o["name_gold"]
                b["slots_ok"] += o["slots_ok"]
                b["slots_tot"] += o["slots_tot"]

            if o["kind"] != "correct":
                errors[o["kind"]] += 1
            if not o["args_exact"]:
                traj_all_exact = False

            sel_items.append((goal, int(o["names_match"]), 1))
            first_items.append((goal, int(o["first_match"]), 1))
            recall_items.append((goal, o["name_hits"], o["name_gold"]))
            exact_items.append((goal, int(o["args_exact"]), 1))
            slot_items.append((goal, o["slots_ok"], o["slots_tot"]))

        completion_items.append((goal, int(traj_all_exact), 1))

    return MultiTurnScore(
        overall={
            **_bucket_dict(overall),
            "tool_selection_ci": cluster_bootstrap(sel_items),
            "first_call_accuracy_ci": cluster_bootstrap(first_items),
            "name_recall_ci": cluster_bootstrap(recall_items),
            "arg_exact_ci": cluster_bootstrap(exact_items),
            "slot_accuracy_ci": cluster_bootstrap(slot_items),
        },
        by_turn_index={str(k): _bucket_dict(v) for k, v in sorted(by_turn.items())},
        by_register={k: _bucket_dict(v) for k, v in sorted(by_register.items())},
        by_script={k: _bucket_dict(v) for k, v in sorted(by_script.items())},
        by_domain={k: _bucket_dict(v) for k, v in sorted(by_domain.items())},
        by_k={str(k): _bucket_dict(v) for k, v in sorted(by_k.items())},
        error_taxonomy=dict(errors),
        task_completion={
            "n": float(len(completion_items)),
            "completion_rate": _rate(
                sum(n for _, n, _ in completion_items), len(completion_items)
            ),
            "completion_ci": cluster_bootstrap(completion_items),
        },
        n_trajectories=len(trajectories),
        n_goals=len({_goal(t) for t in trajectories.values()}),
        status="partial" if missing else "ok",
    )


def score_interactive_rows(
    rows: list[dict],
    trajectories: dict[str, Trajectory],
) -> MultiTurnScore:
    """Score interactive-protocol rows (one row per trajectory).

    Each row carries ``steps`` (the model's predicted call sets in order); they
    are turned into per-step predictions and scored exactly like teacher forcing
    so the two protocols are directly comparable. The difference is that the
    interactive predictions were produced under compounding error.
    """
    predictions: list[StepPrediction] = []
    for row in rows:
        tid = row.get("example_id", "")
        for st in row.get("steps") or []:
            calls = [ToolCall.model_validate(c) for c in (st.get("tool_calls") or [])]
            predictions.append(
                StepPrediction(
                    trajectory_id=tid,
                    step=int(st.get("step", 0)),
                    tool_calls=calls,
                    error=st.get("error"),
                )
            )
    return score_teacher_forced(predictions, trajectories)


def step_predictions_from_rows(rows: list[dict]) -> list[StepPrediction]:
    """Turn teacher-forced JSONL rows back into ``StepPrediction`` objects."""
    out: list[StepPrediction] = []
    for r in rows:
        out.append(
            StepPrediction(
                trajectory_id=r["example_id"],
                step=int(r.get("step", 0)),
                tool_calls=[ToolCall.model_validate(c) for c in (r.get("tool_calls") or [])],
                error=r.get("error"),
            )
        )
    return out


def score_rows(
    protocol: str,
    rows: list[dict],
    trajectories: dict[str, Trajectory],
    only: set[tuple[str, int]] | None = None,
) -> MultiTurnScore:
    """Dispatch to the right scorer for a predictions file's protocol."""
    if protocol == "interactive":
        return score_interactive_rows(rows, trajectories)
    return score_teacher_forced(step_predictions_from_rows(rows), trajectories, only)
