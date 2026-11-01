"""Tests for eval task construction (prompt + gold extraction)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bangla_datasets.eval.task import build_chat_messages, build_tools_param, extract_gold
from bangla_datasets.schema import Trajectory

DATASET = Path("outputs/dataset/bangla_agentic.jsonl")


@pytest.fixture(scope="module")
def traj():
    return Trajectory.model_validate(json.loads(DATASET.read_text().splitlines()[0]))


def test_build_chat_messages_starts_with_system(traj):
    msgs = build_chat_messages(traj)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_build_chat_messages_user_turn_is_the_original(traj):
    msgs = build_chat_messages(traj)
    user = [m for m in msgs if m["role"] == "user"][-1]
    original_first_user = next(m.content for m in traj.messages if m.role.value == "user")
    assert user["content"] == original_first_user


def test_build_tools_param_uses_openai_shape(traj):
    tools = build_tools_param(traj)
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == traj.tools[0].name


def test_extract_gold_returns_first_assistant_tool_call(traj):
    gold = extract_gold(traj)
    expected = next((tc for m in traj.messages if m.tool_calls for tc in m.tool_calls), None)
    if expected:
        assert gold is not None
        assert gold.name == expected.name


def test_extract_gold_none_when_no_tool_call():
    from bangla_datasets.schema import Message, Persona, Role, ToolDef

    traj = Trajectory(
        id="x",
        domain="d",
        persona=Persona(),
        tools=[ToolDef(name="foo", description="d", parameters_json_schema={"type": "object"})],
        system_prompt="s",
        messages=[
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="hello"),
        ],
    )
    assert extract_gold(traj) is None
