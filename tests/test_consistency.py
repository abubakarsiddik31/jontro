"""Tests for stage 2: simulator consistency + language-boundary check.

This is the anti-hallucination gate: each tool output is recomputed via the same
deterministic simulator the orchestrator used, seeded by trajectory metadata, and
byte-compared against the stored tool message content.
"""
from bangla_datasets.schema import Message, Persona, Role, ToolCall, Trajectory
from bangla_datasets.tools.catalog import SEARCH_TRAINS
from bangla_datasets.validation.consistency import check_consistency


def test_consistency_passes_when_simulator_matches() -> None:
    from bangla_datasets.tools.simulators import simulate
    args = {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}
    seed = 7
    t = Trajectory(
        id="c1", domain="railway_booking", persona=Persona(),
        tools=[SEARCH_TRAINS], system_prompt="s",
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="call_1", name="search_trains", arguments=args)]),
            Message(role=Role.TOOL, content=simulate("search_trains", args, seed),
                    tool_call_id="call_1"),
            Message(role=Role.ASSISTANT, content="done"),
        ],
        metadata={"sim_seeds": {"call_1": seed}},
    )
    v = check_consistency(t)
    assert v.passed, v.reasons


def test_consistency_fails_on_phantom_tool_output() -> None:
    t = Trajectory(
        id="c2", domain="railway_booking", persona=Persona(),
        tools=[SEARCH_TRAINS], system_prompt="s",
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="call_1", name="search_trains",
                         arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"})]),
            Message(role=Role.TOOL, content='{"fake": true}', tool_call_id="call_1"),
            Message(role=Role.ASSISTANT, content="done"),
        ],
        metadata={"sim_seeds": {"call_1": 7}},
    )
    v = check_consistency(t)
    assert not v.passed
    assert v.stage_failed == "consistency"


def test_consistency_fails_on_tool_not_in_subset() -> None:
    t = Trajectory(
        id="c3", domain="railway_booking", persona=Persona(),
        tools=[],  # no tools exposed
        system_prompt="s",
        messages=[
            Message(role=Role.USER, content="q"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[
                ToolCall(id="call_1", name="search_trains",
                         arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"})]),
            Message(role=Role.TOOL, content="{}", tool_call_id="call_1"),
            Message(role=Role.ASSISTANT, content="done"),
        ],
        metadata={"sim_seeds": {"call_1": 7}},
    )
    v = check_consistency(t)
    assert not v.passed
