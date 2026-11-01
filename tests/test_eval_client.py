"""Tests for the OpenAI-compatible eval client (no network)."""
from __future__ import annotations

import json

from bangla_datasets.eval.client import Prediction, parse_tool_calls


def test_parse_openai_tool_calls_with_string_arguments():
    """openai >=1.x returns arguments as a JSON string; we parse to dict."""
    raw = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "search_trains",
                "arguments": json.dumps({"from": "ঢাকা", "to": "চট্টগ্রাম"}),
            },
        }
    ]
    calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "search_trains"
    assert calls[0].arguments == {"from": "ঢাকা", "to": "চট্টগ্রাম"}
    assert calls[0].id == "call_abc"


def test_parse_empty_when_none():
    assert parse_tool_calls(None) == []
    assert parse_tool_calls([]) == []


def test_parse_invalid_json_arguments_keeps_call_with_empty_args():
    """Invalid args => keep the call (so scoring flags it arg-invalid), empty args."""
    raw = [
        {
            "id": "call_x",
            "type": "function",
            "function": {"name": "foo", "arguments": "not-json{("},
        }
    ]
    calls = parse_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "foo"
    assert calls[0].arguments == {}


def test_prediction_carries_error():
    p = Prediction(tool_calls=[], raw_text="", error="timeout")
    assert p.error == "timeout"
    assert p.tool_call is None
