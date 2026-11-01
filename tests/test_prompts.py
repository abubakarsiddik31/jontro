from bangla_datasets.gemini.prompts import (
    build_assistant_prompt,
    build_judge_prompt,
    build_persona_prompt,
)
from bangla_datasets.schema import Persona

_JUDGE_DIMENSIONS = [
    "task_completion",
    "tool_correctness",
    "bangla_fluency",
    "code_switching",
    "coherence",
]


def test_persona_prompt_includes_register() -> None:
    p = build_persona_prompt(Persona(register="apni"), "railway_booking", "Book a ticket")
    assert "apni" in p.lower() or "আপনি" in p


def test_assistant_prompt_mentions_language_boundary() -> None:
    p = build_assistant_prompt([{"name": "f", "description": "d"}])
    assert "bangla" in p.lower() or "বাংলা" in p.lower()


def test_judge_prompt_lists_five_dimensions() -> None:
    p = build_judge_prompt()
    for d in _JUDGE_DIMENSIONS:
        assert d in p
