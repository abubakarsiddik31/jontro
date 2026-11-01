"""Tests for stage 1: schema & structural checks (pure Python)."""
import pytest
from pydantic import ValidationError

from bangla_datasets.schema import Message, Persona, Role, Trajectory
from bangla_datasets.validation.schema_check import check_schema


def _good_trajectory() -> Trajectory:
    return Trajectory(
        id="t1", domain="railway_booking", persona=Persona(),
        tools=[], system_prompt="s",
        messages=[Message(role=Role.USER, content="hi"),
                  Message(role=Role.ASSISTANT, content="hello")],
    )


def test_schema_passes_good_trajectory() -> None:
    v = check_schema(_good_trajectory())
    assert v.passed is True


def test_schema_fails_missing_tool_call_id() -> None:
    # The tool role requires tool_call_id — enforced by the schema's
    # model_validator at construction time. We build inside pytest.raises so the
    # ValidationError is captured.
    with pytest.raises(ValidationError):
        Trajectory(
            id="t2", domain="d", persona=Persona(), tools=[], system_prompt="s",
            messages=[Message(role=Role.USER, content="x"),
                      Message(role=Role.TOOL, content="r")],  # missing tool_call_id
        )
