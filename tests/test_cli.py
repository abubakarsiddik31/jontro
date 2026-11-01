"""CLI smoke tests.

The CLI's runtime paths (generate/validate/publish) require a live Gemini
client, so we only assert that the argparse surface is wired correctly: each
subcommand's `--help` must exit cleanly with code 0. No network calls.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_cli_generate_help_runs() -> None:
    from bangla_datasets.cli import main

    with patch.object(sys, "argv", ["bangla-datasets", "generate", "--help"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0


def test_cli_validate_help_runs() -> None:
    from bangla_datasets.cli import main

    with patch.object(sys, "argv", ["bangla-datasets", "validate", "--help"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0


def test_cli_publish_help_runs() -> None:
    from bangla_datasets.cli import main

    with patch.object(sys, "argv", ["bangla-datasets", "publish", "--help"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0


def test_cmd_generate_with_fake_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration test: the generate path writes trajectories incrementally to
    JSONL using a fake client (no network). Guards FIX 2 (per-trajectory
    isolation), FIX 3 (id alignment), and FIX 4 (incremental write)."""
    from bangla_datasets.gemini.client import AssistantResponse

    class FakeClient:
        """Mimics the GeminiClient surface the orchestrator drives."""

        def __init__(self, **kwargs: object) -> None:
            pass

        def persona_turn(self, system: str, history: list[object], seed: int, **kw: object) -> str:
            return "ঢাকা থেকে চট্টগ্রাম ট্রেন চাই।"

        def assistant_turn(
            self, system: str, history: list[object], tools: list[object], seed: int,
        ) -> AssistantResponse:
            # Final answer, no tool calls -> trajectory terminates after one turn.
            return AssistantResponse(text="ঠিক আছে, ৩টি ট্রেন পাওয়া গেছে।", tool_calls=[])

    import bangla_datasets.cli as cli_module
    monkeypatch.setattr(cli_module, "GeminiClient", FakeClient)

    from bangla_datasets.cli import main
    run_dir = tmp_path / "test_run"
    with patch.object(
        sys, "argv",
        [
            "bangla-datasets", "generate",
            "--recipe", "recipes/agentic_v1.yaml",
            "--run-dir", str(run_dir),
            "--count", "3",
        ],
    ):
        rc = main()

    assert rc == 0
    traj_file = run_dir / "trajectories.jsonl"
    assert traj_file.exists()
    lines = traj_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_cmd_generate_skips_bad_trajectory_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 2: a hallucinated tool name (or any orchestrator error) must skip the
    trajectory and continue the batch, not abort it. The bad trajectory is NOT
    marked done (so it regenerates on resume), and good trajectories still
    land in the output."""
    from bangla_datasets.gemini.client import AssistantResponse
    from bangla_datasets.schema import ToolCall

    class FlakyClient:
        """First assistant call hallucinates a tool name; later calls finalize."""

        def __init__(self, **kwargs: object) -> None:
            self._n = 0

        def persona_turn(self, system: str, history: list[object], seed: int, **kw: object) -> str:
            return "যাত্রা শুরু করুন।"

        def assistant_turn(
            self, system: str, history: list[object], tools: list[object], seed: int,
        ) -> AssistantResponse:
            self._n += 1
            if self._n == 1:
                # Hallucinated tool name not in the registry -> KeyError in simulate.
                return AssistantResponse(
                    text="",
                    tool_calls=[ToolCall(id="c1", name="nonexistent_tool", arguments={})],
                )
            return AssistantResponse(text="ঠিক আছে।", tool_calls=[])

    import bangla_datasets.cli as cli_module
    monkeypatch.setattr(cli_module, "GeminiClient", FlakyClient)

    from bangla_datasets.cli import main
    run_dir = tmp_path / "flaky_run"
    with patch.object(
        sys, "argv",
        [
            "bangla-datasets", "generate",
            "--recipe", "recipes/agentic_v1.yaml",
            "--run-dir", str(run_dir),
            "--count", "2",
        ],
    ):
        rc = main()

    assert rc == 0  # batch completed despite the failure
    traj_file = run_dir / "trajectories.jsonl"
    lines = traj_file.read_text(encoding="utf-8").strip().splitlines()
    # Trajectory 0 failed (hallucinated tool name) and is NOT written; trajectory
    # 1 succeeded and IS written. This proves the batch continued past the error
    # and that failed trajectories are skipped, not aborted.
    assert len(lines) == 1
    cp_file = run_dir / "checkpoint.json"
    done = json.loads(cp_file.read_text())["done"]
    assert done == ["bd-agentic-000001"]  # the failed traj-000000 is not marked done
