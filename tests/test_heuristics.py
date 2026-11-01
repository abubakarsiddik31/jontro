"""Tests for stage 4: heuristic filters (length, repetition, language-mix)."""
from bangla_datasets.schema import Message, Persona, Role, ToolCall, Trajectory
from bangla_datasets.utils.script import cross_register_evidence
from bangla_datasets.validation.heuristics import (
    contains_sadhu_bhasha,
    register_conformance,
    run_heuristics,
    validate_multi_turn_trajectory,
)


def _traj(messages: list[Message]) -> Trajectory:
    return Trajectory(id="h1", domain="d", persona=Persona(), tools=[], system_prompt="s",
                      messages=messages)


def _tc(i: int) -> ToolCall:
    return ToolCall(
        id=f"c{i}", name="search_trains",
        arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"},
    )


def test_heuristics_pass_normal() -> None:
    v = run_heuristics(_traj([
        Message(role=Role.USER, content="একটি প্রশ্ন?"),
        Message(role=Role.ASSISTANT, content="একটি সাধারণ উত্তর এখানে।"),
    ]))
    assert v.passed


def test_heuristics_fail_tiny_turn() -> None:
    v = run_heuristics(_traj([
        Message(role=Role.USER, content="x"),  # too short
        Message(role=Role.ASSISTANT, content="ok answer"),
    ]))
    assert not v.passed


def test_heuristics_fail_repeated_call() -> None:
    v = run_heuristics(_traj([
        Message(role=Role.USER, content="question here please"),
        Message(role=Role.ASSISTANT, content="thinking", tool_calls=[_tc(1), _tc(2), _tc(3)]),
    ]))
    assert not v.passed


def test_contains_sadhu_bhasha() -> None:
    assert contains_sadhu_bhasha("মহাশয়, আমি আপনার নিকট তথ্য জানতে ইচ্ছুক।") is True
    assert contains_sadhu_bhasha("ঢাকা হইতে চট্টগ্রাম যাইবে।") is True
    assert contains_sadhu_bhasha("উক্ত বিষয়টি বিবেচনা করুন।") is True

    # Modern Chalito MUST pass
    assert contains_sadhu_bhasha("ঢাকা থেকে চট্টগ্রাম যাওয়ার ট্রেন দেখতে চাই।") is False
    assert contains_sadhu_bhasha("আমি রাজশাহী অঞ্চলের আবহাওয়া জানতে চাচ্ছিলাম।") is False


def test_contains_sadhu_bhasha_expanded_families() -> None:
    """The hardened detector covers -ইয়া, -ইল, -ইব- and -ইয়াছ- families."""
    assert contains_sadhu_bhasha("আমি করিয়া দিব।") is True
    assert contains_sadhu_bhasha("সে আসিয়া বলিল।") is True
    assert contains_sadhu_bhasha("তাহার সহিত গমন করিব।") is True
    # Chalito must still pass
    assert contains_sadhu_bhasha("আমি এটা করে দেব।") is False
    assert contains_sadhu_bhasha("আমি সেখানে যাব এবং দেখব।") is False


# --- Register conformance with verb morphology -------------------------------

def test_cross_register_evidence_separates_marker_types() -> None:
    ev = cross_register_evidence("তুমি কি আমার হয়ে বুক করবে", "apni")
    assert "তুমি" in ev["pronouns"]
    assert ev["verbs"]  # করবে carries the familiar verb suffix


def test_cross_register_pronoun_flagged() -> None:
    # Persona default register is apni; a তুমি pronoun contradicts it.
    traj = _traj([Message(role=Role.USER, content="তুমি আমার জন্য একটা ট্রেনের টিকিট বুক করে দাও")])
    c = register_conformance(traj)
    assert c["pronoun_conflict"] is True
    assert "তুমি" in c["pronouns"]


def test_register_verb_evidence_is_advisory_not_a_pronoun() -> None:
    # 'অফিস' ends in a tui-ish suffix; the suffix match is advisory, and never
    # masquerades as a pronoun conflict.
    ev = cross_register_evidence("আমি অফিসে যাবো কেমন হবে বলুন", "tui")
    assert ev["pronouns"] == []


def test_apni_turn_is_register_clean() -> None:
    traj = _traj([Message(role=Role.USER, content="আপনি আমার জন্য একটি ট্রেনের টিকিট বুক করে দেবেন প্লিজ")])
    c = register_conformance(traj)
    assert c["pronoun_conflict"] is False


