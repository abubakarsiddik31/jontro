"""Tests for stage 3: LLM judge (tested with a FAKE client — no network)."""
from bangla_datasets.schema import JudgeScore, Message, Persona, Role, Trajectory, Verdict
from bangla_datasets.validation.judge import run_judge


class FakeJudgeClient:
    """Stub client implementing the `.judge(traj_json, rubric) -> Verdict` protocol."""

    def judge(self, trajectory_json: str, rubric_prompt: str) -> Verdict:
        return Verdict(passed=True, scores=[
            JudgeScore(dimension="task_completion", score=5, reason="ok"),
            JudgeScore(dimension="tool_correctness", score=4, reason="ok"),
            JudgeScore(dimension="bangla_fluency", score=4, reason="ok"),
            JudgeScore(dimension="code_switching", score=5, reason="ok"),
            JudgeScore(dimension="coherence", score=4, reason="ok"),
        ])


def test_run_judge_uses_client() -> None:
    t = Trajectory(id="j1", domain="d", persona=Persona(), tools=[], system_prompt="s",
                   messages=[Message(role=Role.USER, content="q"),
                             Message(role=Role.ASSISTANT, content="a")])
    v = run_judge(t, FakeJudgeClient(), pass_threshold=4)  # type: ignore[arg-type]
    assert v.passed is True
    assert len(v.scores) == 5
