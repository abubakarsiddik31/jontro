"""Tests for the new variation axes: conversation script (bengali|banglish) and
system-prompt language (bangla|english).

These cover the highest-risk change: heuristics.py must NOT reject a legitimate
Banglish (Romanized) turn as a "language flip", but MUST still reject a Bengali
trajectory that drops into pure English.
"""
from __future__ import annotations

from bangla_datasets.gemini.prompts import (
    build_assistant_prompt,
    build_judge_prompt,
    build_persona_prompt,
)
from bangla_datasets.schema import Message, Persona, Role, ToolCall, Trajectory
from bangla_datasets.tools.catalog import GET_LIVE_SCORE, SEARCH_RESTAURANTS
from bangla_datasets.validation.heuristics import run_heuristics


def _traj(content: str, persona: Persona, domain: str = "sports_scores") -> Trajectory:
    """Build a minimal trajectory with one user turn of *content*."""
    return Trajectory(
        id="test-traj",
        domain=domain,
        persona=persona,
        tools=[GET_LIVE_SCORE],
        system_prompt="x",
        messages=[Message(role=Role.USER, content=content)],
    )


# --- Persona validators -------------------------------------------------------

def test_persona_valid_scripts() -> None:
    assert Persona(script="bengali").script == "bengali"
    assert Persona(script="banglish").script == "banglish"


def test_persona_default_script_is_bengali() -> None:
    """Old data without the field defaults to bengali (backward compat)."""
    assert Persona().script == "bengali"
    assert Persona().system_prompt_language == "bangla"


def test_persona_rejects_bad_script() -> None:
    import pytest
    with pytest.raises(ValueError):
        Persona(script="romanized")
    with pytest.raises(ValueError):
        Persona(system_prompt_language="french")


# --- The critical Banglish heuristics gate -----------------------------------

def test_banglish_turn_passes_when_script_is_banglish() -> None:
    """A Romanized-Bangla turn must pass when persona.script == banglish."""
    p = Persona(script="banglish")
    traj = _traj("ami ajker cricket match er live score jante chai", p)
    assert run_heuristics(traj).passed


def test_banglish_turn_fails_when_script_is_bengali() -> None:
    """A no-Bengali-script turn must FAIL when persona.script == bengali."""
    p = Persona(script="bengali")
    traj = _traj("ami ajker cricket match er live score jante chai", p)
    assert not run_heuristics(traj).passed


def test_bengali_turn_passes_when_script_is_bengali() -> None:
    p = Persona(script="bengali")
    traj = _traj("আমি আজকের ক্রিকেট ম্যাচের লাইভ স্কোর জানতে চাই।", p)
    assert run_heuristics(traj).passed


def test_bengali_turn_fails_when_script_is_banglish() -> None:
    """A Bengali-script turn in a banglish trajectory is a script flip."""
    p = Persona(script="banglish")
    traj = _traj("আমি আজকের ক্রিকেট ম্যাচের লাইভ স্কোর জানতে চাই।", p)
    assert not run_heuristics(traj).passed


def test_short_turns_not_script_checked() -> None:
    """Turns <= 20 chars are length-only, not script-checked."""
    p = Persona(script="bengali")
    traj = _traj("হ্যাঁ", p)  # valid short Bengali
    v = run_heuristics(traj)
    assert v.passed


# --- Prompt builders ----------------------------------------------------------

def test_assistant_prompt_english_variant() -> None:
    prompt = build_assistant_prompt(
        [{"name": "get_live_score", "description": "x"}], language="english"
    )
    assert "You are a helpful AI assistant" in prompt
    # Tool schema still present
    assert "get_live_score" in prompt


def test_assistant_prompt_bangla_variant() -> None:
    prompt = build_assistant_prompt(
        [{"name": "get_live_score", "description": "x"}], language="bangla"
    )
    assert "সহায়ক এআই সহকারী" in prompt


def test_persona_prompt_banglish_variant() -> None:
    p = Persona(script="banglish", register="tumi")
    prompt = build_persona_prompt(p, "sports_scores", "[test] score")
    # Banglish persona prompt is Romanized, not Bengali script
    assert "banglish" in prompt.lower() or "roman" in prompt.lower()


def test_persona_prompt_bengali_variant() -> None:
    p = Persona(script="bengali", register="apni")
    prompt = build_persona_prompt(p, "sports_scores", "[test] স্কোর")
    # Bengali persona prompt contains Bengali
    assert "সাধারণ" in prompt


def test_judge_prompt_keeps_5_dimensions_banglish() -> None:
    """Dimension names must stay fixed so judge.py len(scores)==5 holds."""
    prompt = build_judge_prompt(script="banglish")
    for dim in ("task_completion", "tool_correctness", "bangla_fluency",
                "code_switching", "coherence"):
        assert dim in prompt


def test_judge_prompt_keeps_5_dimensions_bengali() -> None:
    prompt = build_judge_prompt(script="bengali")
    for dim in ("task_completion", "tool_correctness", "bangla_fluency",
                "code_switching", "coherence"):
        assert dim in prompt


# --- New domain tools wired correctly ----------------------------------------

def test_new_domains_have_tools_and_simulators() -> None:
    from bangla_datasets.tools.catalog import DOMAIN_TOOLS
    from bangla_datasets.tools.simulators import _SIMULATORS

    for domain, tools in DOMAIN_TOOLS.items():
        assert len(tools) >= 1, f"{domain} has no tools"
        for t in tools:
            assert t.name in _SIMULATORS, f"{t.name} ({domain}) has no simulator"
