"""Tests for the eval runner using a fake client (no network)."""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.eval.client import Prediction
from bangla_datasets.eval.runner import run_model
from bangla_datasets.schema import Message, Persona, Role, ToolCall, ToolDef, Trajectory


def _traj(i: int) -> Trajectory:
    return Trajectory(
        id=f"t-{i:03d}",
        domain="d",
        persona=Persona(),
        tools=[
            ToolDef(
                name="search_trains",
                description="d",
                parameters_json_schema={"type": "object"},
            )
        ],
        system_prompt="s",
        messages=[Message(role=Role.USER, content="hi")],
    )


class FakeClient:
    def __init__(self) -> None:
        self.model = "fake-model"

    def predict(self, traj: Trajectory) -> Prediction:  # noqa: ANN001
        if traj.id.endswith("002"):
            return Prediction(tool_calls=[], raw_text="", error="boom")
        return Prediction(
            tool_calls=[ToolCall(id="p", name="search_trains", arguments={})],
            raw_text="",
        )


def test_run_streams_predictions(tmp_path: Path):
    out = tmp_path / "preds.jsonl"
    report = run_model(FakeClient(), [_traj(i) for i in range(5)], out, min_delay=0)
    assert report.model == "fake-model"
    assert report.n == 5
    assert report.n_errors == 1
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 5
    assert rows[2]["error"] == "boom"


def test_run_is_resumable(tmp_path: Path):
    """If the output already has some rows, the runner skips those ids."""
    out = tmp_path / "preds.jsonl"
    out.write_text(
        json.dumps(
            {"example_id": "t-000", "model": "fake-model", "tool_call": None, "error": None}
        )
        + "\n"
    )
    report = run_model(FakeClient(), [_traj(i) for i in range(3)], out, min_delay=0)
    assert report.n == 3
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3  # only 2 new appended; t-000 already present
