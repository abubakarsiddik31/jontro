"""Tests for output writers: canonical JSONL, ShareGPT, and verdicts."""
import json
from pathlib import Path

from bangla_datasets.schema import Message, Persona, Role, Trajectory
from bangla_datasets.writers.jsonl import write_jsonl, write_verdicts
from bangla_datasets.writers.openai_format import to_openai_format, write_openai_jsonl
from bangla_datasets.writers.sharegpt import write_sharegpt


def _traj() -> Trajectory:
    return Trajectory(id="w1", domain="d", persona=Persona(), tools=[], system_prompt="sys",
                      messages=[Message(role=Role.USER, content="প্রশ্ন?"),
                                Message(role=Role.ASSISTANT, content="উত্তর।")])


def test_jsonl_writer_one_line_per_traj(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    write_jsonl([_traj(), _traj()], p)
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    obj = json.loads(lines[0])
    assert obj["id"] == "w1"
    assert "messages" in obj


def test_sharegpt_writer_format(tmp_path: Path) -> None:
    p = tmp_path / "sg.jsonl"
    write_sharegpt([_traj()], p)
    obj = json.loads(p.read_text().splitlines()[0])
    assert "conversations" in obj
    roles = [c.get("from") for c in obj["conversations"]]
    assert "human" in roles and "gpt" in roles


def test_verdicts_writer(tmp_path: Path) -> None:
    from bangla_datasets.schema import Verdict

    p = tmp_path / "v.jsonl"
    write_verdicts([Verdict(trajectory_id="w1", passed=True)], p)
    assert json.loads(p.read_text().splitlines()[0])["passed"] is True


def test_openai_format_writer(tmp_path: Path) -> None:
    p = tmp_path / "openai.jsonl"
    write_openai_jsonl([_traj()], p)
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "messages" in obj
    assert obj["messages"][0]["role"] == "system"
    assert obj["messages"][1]["role"] == "user"
    assert obj["messages"][2]["role"] == "assistant"

