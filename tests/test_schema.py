import pytest
from pydantic import ValidationError

from bangla_datasets.schema import (
    JudgeScore,
    Message,
    Persona,
    Role,
    ToolCall,
    ToolDef,
    Trajectory,
    Verdict,
)


def test_tooldef_minimal() -> None:
    t = ToolDef(
        name="search_trains",
        description="Search trains.",
        parameters_json_schema={"type": "object", "properties": {}, "required": []},
    )
    assert t.name == "search_trains"


def test_tooldef_rejects_non_english_name() -> None:
    # Tool/function names must be English/Latin script (language boundary)
    with pytest.raises(ValidationError):
        ToolDef(
            name="ট্রেন_খুঁজুন",
            description="x",
            parameters_json_schema={"type": "object", "properties": {}, "required": []},
        )


def test_toolcall_valid() -> None:
    tc = ToolCall(id="call_1", name="search_trains", arguments={"from": "ঢাকা"})
    assert tc.arguments["from"] == "ঢাকা"


def test_message_tool_role_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError):
        Message(role=Role.TOOL, content="result")


def test_message_assistant_with_tool_calls() -> None:
    m = Message(
        role=Role.ASSISTANT,
        content="thinking",
        tool_calls=[ToolCall(id="c1", name="f", arguments={})],
    )
    assert m.tool_calls[0].id == "c1"


def test_persona_defaults() -> None:
    p = Persona()
    assert p.register in {"tui", "tumi", "apni"}


def test_trajectory_message_ordering() -> None:
    with pytest.raises(ValidationError):
        Trajectory(
            id="t1",
            domain="d",
            persona=Persona(),
            tools=[],
            system_prompt="s",
            messages=[Message(role=Role.ASSISTANT, content="hi")],
        )


def test_judgescore_bounds() -> None:
    with pytest.raises(ValidationError):
        JudgeScore(dimension="x", score=6, reason="r")


def test_tooldef_rejects_bangla_key_in_nested_list() -> None:
    # Regression: Bangla keys inside JSON-schema list constructs (anyOf, prefixItems, items)
    # must be caught by the recursive language-boundary check.
    with pytest.raises(ValidationError):
        ToolDef(name="x", description="x", parameters_json_schema={
            "prefixItems": [{"type": "object", "properties": {"বাংলা": {"type": "string"}}}]
        })


def test_tooldef_rejects_bangla_key_in_anyof() -> None:
    with pytest.raises(ValidationError):
        ToolDef(name="y", description="y", parameters_json_schema={
            "anyOf": [{"type": "object", "properties": {"নাম": {"type": "string"}}}]
        })


def test_verdict_pass_when_all_scores_above_threshold() -> None:
    v = Verdict(
        passed=True,
        scores=[
            JudgeScore(dimension="task_completion", score=4, reason="ok"),
            JudgeScore(dimension="tool_correctness", score=5, reason="ok"),
            JudgeScore(dimension="bangla_fluency", score=4, reason="ok"),
            JudgeScore(dimension="code_switching", score=4, reason="ok"),
            JudgeScore(dimension="coherence", score=5, reason="ok"),
        ],
    )
    assert v.passed is True
