"""Tests for the multi-turn eval harness (task, scoring, runner, parquet codec).

No network: the runner is exercised through its ``oracle`` path, which echoes the
gold calls and therefore must score 100% on every metric.
"""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.eval.multiturn_runner import run_interactive, run_teacher_forced
from bangla_datasets.eval.multiturn_scoring import (
    MultiTurnScore,
    StepPrediction,
    cluster_bootstrap,
    norm_value,
    score_interactive_rows,
    score_step,
    score_teacher_forced,
    step_predictions_from_rows,
)
from bangla_datasets.eval.multiturn_task import (
    build_step_messages,
    iter_decision_points,
    message_to_openai,
    simulate_seed,
)
from bangla_datasets.schema import Message, Persona, Role, ToolCall, ToolDef, Trajectory
from bangla_datasets.splits.parquet_codec import decode_row, encode_row


def _tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="search_trains",
            description="d",
            parameters_json_schema={
                "type": "object",
                "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
            },
        ),
        ToolDef(
            name="get_weather",
            description="d",
            parameters_json_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        ),
    ]


def _traj(tid: str = "t1", goal: str = "g1", domain: str = "d") -> Trajectory:
    """Two decision points; step 0 has one call, step 1 has one call."""
    return Trajectory(
        id=tid,
        domain=domain,
        persona=Persona(register="apni", script="bengali"),
        tools=_tools(),
        system_prompt="sys",
        messages=[
            Message(role=Role.USER, content="প্রথম প্রশ্ন?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_trains",
                        arguments={"from": "ঢাকা", "to": "চট্টগ্রাম"},
                    )
                ],
            ),
            Message(role=Role.TOOL, tool_call_id="c1", content='{"ok": true}'),
            Message(role=Role.ASSISTANT, content="প্রথম উত্তর।"),
            Message(role=Role.USER, content="দ্বিতীয় প্রশ্ন?"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="c2", name="get_weather", arguments={"location": "খুলনা"})],
            ),
            Message(role=Role.TOOL, tool_call_id="c2", content='{"temp": 25}'),
            Message(role=Role.ASSISTANT, content="দ্বিতীয় উত্তর।"),
        ],
        metadata={"persona_goal": goal},
    )


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------
def test_iter_decision_points_pairs_user_turns() -> None:
    dps = iter_decision_points(_traj())
    assert [dp.index for dp in dps] == [0, 1]
    assert [c.name for c in dps[0].gold_calls] == ["search_trains"]
    assert [c.name for c in dps[1].gold_calls] == ["get_weather"]
    # each step is preceded by exactly one user turn
    assert [len(dp.pending_user) for dp in dps] == [1, 1]
    assert dps[1].pending_user[0].content == "দ্বিতীয় প্রশ্ন?"


def test_build_step_messages_prefix_is_gold_and_excludes_target() -> None:
    traj = _traj()
    dp0, dp1 = iter_decision_points(traj)
    m0 = build_step_messages(traj, dp0)
    assert [m["role"] for m in m0] == ["system", "user"]

    m1 = build_step_messages(traj, dp1)
    assert [m["role"] for m in m1] == ["system", "user", "assistant", "tool", "assistant", "user"]
    # the target step's own call must not appear in its prompt
    assert all("get_weather" not in json.dumps(m) for m in m1)


def test_message_to_openai_arguments_are_json_strings_and_signature_dropped() -> None:
    tc = ToolCall(
        id="c1",
        name="search_trains",
        arguments={"from": "ঢাকা"},
        thought_signature_b64="AAAA",
    )
    msg = message_to_openai(Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]))
    call = msg["tool_calls"][0]
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"from": "ঢাকা"}
    assert "thought_signature_b64" not in json.dumps(msg)


def test_simulate_seed_is_deterministic_and_position_dependent() -> None:
    a = simulate_seed("t1", 0, 0)
    assert a == simulate_seed("t1", 0, 0)
    assert a != simulate_seed("t1", 0, 1)
    assert a != simulate_seed("t1", 1, 0)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def test_norm_value_unifies_bengali_and_ascii_digits() -> None:
    assert norm_value("৯৯০০৯৯১১২২") == norm_value(9900991122)
    assert norm_value(" Dhaka ") == norm_value("dhaka")


def _calls(name: str, **args: object) -> list[ToolCall]:
    return [ToolCall(id="x", name=name, arguments=dict(args))]


def test_score_step_correct_and_null_padding_ignored() -> None:
    gold = _calls("search_trains", **{"from": "ঢাকা", "to": "চট্টগ্রাম"})
    pred = _calls("search_trains", **{"from": "ঢাকা", "to": "চট্টগ্রাম", "seat_class": None})
    o = score_step(pred, gold)
    assert o["kind"] == "correct"
    assert o["args_exact"] and o["names_match"]
    assert o["slots_tot"] == 2 and o["slots_ok"] == 2


def test_score_step_wrong_tool() -> None:
    o = score_step(_calls("get_weather", location="খুলনা"), _calls("search_trains", **{"from": "a"}))
    assert o["kind"] == "wrong_tool"
    assert not o["names_match"] and not o["args_exact"]


def test_score_step_wrong_arg_values() -> None:
    o = score_step(
        _calls("search_trains", **{"from": "ঢাকা", "to": "সিলেট"}),
        _calls("search_trains", **{"from": "ঢাকা", "to": "চট্টগ্রাম"}),
    )
    assert o["kind"] == "wrong_arg_values"
    assert o["names_match"] and not o["args_exact"]
    assert o["slots_ok"] == 1 and o["slots_tot"] == 2


def test_score_step_no_call() -> None:
    o = score_step([], _calls("search_trains", **{"from": "a"}))
    assert o["kind"] == "no_call"
    assert o["slots_tot"] == 1


def test_score_step_parallel_calls_are_order_insensitive() -> None:
    gold = [
        ToolCall(id="a", name="search_trains", arguments={"from": "ঢাকা"}),
        ToolCall(id="b", name="get_weather", arguments={"location": "খুলনা"}),
    ]
    pred = list(reversed(gold))
    o = score_step(pred, gold)
    assert o["kind"] == "correct"


def test_score_step_first_call_and_recall_are_separate_from_set_match() -> None:
    """Set equality, first-call accuracy and recall are three different questions.

    A reordered but complete prediction is set-correct and recovers every gold
    name, yet its first call differs from the gold first call; a prediction that
    drops one of two gold calls is set-wrong and only half-recalled.
    """
    gold = [
        ToolCall(id="a", name="search_trains", arguments={"from": "ঢাকা"}),
        ToolCall(id="b", name="get_weather", arguments={"location": "খুলনা"}),
    ]
    reordered = score_step(list(reversed(gold)), gold)
    assert reordered["names_match"] is True
    assert reordered["first_match"] is False  # order is not the gold order
    assert reordered["name_hits"] == 2 and reordered["name_gold"] == 2

    dropped = score_step([gold[1]], gold)
    assert dropped["names_match"] is False
    assert dropped["first_match"] is False  # get_weather is not the gold first call
    assert dropped["name_hits"] == 1 and dropped["name_gold"] == 2


def test_cluster_bootstrap_needs_three_goals() -> None:
    assert cluster_bootstrap([("g1", 1, 1), ("g2", 1, 1)]) is None
    ci = cluster_bootstrap([(f"g{i}", 1, 1) for i in range(5)])
    assert ci is not None and ci[0] <= 100.0 <= ci[1]


def test_score_teacher_forced_aggregates_and_completes() -> None:

    trajs = {t.id: t for t in (_traj("t1", "g1"), _traj("t2", "g2"), _traj("t3", "g3"))}
    preds = [
        StepPrediction(t.id, dp.index, dp.gold_calls)
        for t in trajs.values()
        for dp in iter_decision_points(t)
    ]
    s = score_teacher_forced(preds, trajs)
    assert isinstance(s, MultiTurnScore)
    assert s.status == "ok"
    assert s.overall["tool_selection_rate"] == 1.0
    assert s.overall["first_call_accuracy"] == 1.0
    assert s.overall["name_recall"] == 1.0
    assert s.overall["arg_exact_rate"] == 1.0
    assert s.task_completion["completion_rate"] == 1.0
    assert str(0) in s.by_turn_index and str(1) in s.by_turn_index


def test_score_teacher_forced_missing_prediction_is_no_call_and_partial() -> None:
    trajs = {t.id: t for t in (_traj("t1", "g1"), _traj("t2", "g2"), _traj("t3", "g3"))}
    preds = [StepPrediction("t1", 0, iter_decision_points(_traj("t1", "g1"))[0].gold_calls)]
    s = score_teacher_forced(preds, trajs)
    assert s.status == "partial"
    assert s.error_taxonomy.get("no_call", 0) == 5  # t1 step1 + t2/t3 steps 0,1


# --------------------------------------------------------------------------
# runner (oracle path — no network)
# --------------------------------------------------------------------------
def test_teacher_forced_oracle_scores_100_and_is_resumable(tmp_path: Path) -> None:
    trajs = [_traj("t1", "g1"), _traj("t2", "g2"), _traj("t3", "g3")]
    out = tmp_path / "tf.jsonl"
    r = run_teacher_forced(None, trajs, out, min_delay=0, oracle=True)
    assert r.n == 6  # 2 decision points x 3 trajs
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    s = score_teacher_forced(step_predictions_from_rows(rows), {t.id: t for t in trajs})
    assert s.overall["arg_exact_rate"] == 1.0
    # re-run appends nothing
    run_teacher_forced(None, trajs, out, min_delay=0, oracle=True)
    assert len(out.read_text().splitlines()) == 6


def test_interactive_oracle_runs_and_scores_100(tmp_path: Path) -> None:
    trajs = [_traj("t1", "g1"), _traj("t2", "g2"), _traj("t3", "g3")]
    out = tmp_path / "int.jsonl"
    r = run_interactive(None, trajs, out, min_delay=0, oracle=True)
    assert r.n == 3
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["protocol"] == "interactive" for r in rows)
    assert all(len(r["steps"]) == 2 for r in rows)

    s = score_interactive_rows(rows, {t.id: t for t in trajs})
    assert s.overall["arg_exact_rate"] == 1.0
    assert s.task_completion["completion_rate"] == 1.0


# --------------------------------------------------------------------------
# parquet codec
# --------------------------------------------------------------------------
def test_parquet_codec_roundtrip_preserves_open_fields() -> None:
    row = {
        "id": "t1",
        "tools": [
            {"name": "t", "parameters_json_schema": {"type": "object", "properties": {"a": {}}}}
        ],
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "c", "name": "t", "arguments": {"a": 1}}]}
        ],
    }
    encoded = encode_row(row)
    assert isinstance(encoded["tools"][0]["parameters_json_schema"], str)
    assert isinstance(encoded["messages"][0]["tool_calls"][0]["arguments"], str)
    assert decode_row(encoded) == row


def test_shipped_parquet_roundtrips_to_released_jsonl() -> None:
    """Regression: the Parquet release must not pad tool args with nulls."""
    import pytest

    from bangla_datasets.splits.io_parquet import load_split_parquet

    jl = Path("outputs/release_multiturn/jontro_multiturn_2000.jsonl")
    pq = Path("outputs/parquet/jontro_test.parquet")
    if not (jl.exists() and pq.exists()):
        pytest.skip("release artifacts not present")
    src = {json.loads(line)["id"]: json.loads(line) for line in jl.read_text().splitlines()}
    differ = 0
    for row in load_split_parquet(pq):
        if row["id"] in src and row["messages"] != src[row["id"]]["messages"]:
            differ += 1
    assert differ == 0
