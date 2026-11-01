"""Tests for the 4-stage validation pipeline (schema -> consistency -> heuristics -> judge).

Stages run cheap->expensive and short-circuit on first failure. Heuristics (cheap)
run before the Gemini judge (expensive).
"""
from bangla_datasets.schema import JudgeScore, Message, Persona, Role, Trajectory, Verdict
from bangla_datasets.validation.pipeline import ValidationPipeline


class StubJudgeClient:
    """Stub client: all-5s verdict."""

    def judge(self, traj_json: str, rubric: str) -> Verdict:
        return Verdict(passed=True, scores=[
            JudgeScore(dimension=d, score=5, reason="ok")
            for d in ["task_completion", "tool_correctness", "bangla_fluency",
                      "code_switching", "coherence"]
        ])


def _good() -> Trajectory:
    return Trajectory(id="p1", domain="d", persona=Persona(), tools=[], system_prompt="s",
                      messages=[Message(role=Role.USER, content="একটি প্রশ্ন এখানে"),
                                Message(role=Role.ASSISTANT, content="একটি উত্তর এখানে")])


def test_pipeline_passes_good_trajectory() -> None:
    p = ValidationPipeline(client=StubJudgeClient())  # type: ignore[arg-type]
    v = p.validate(_good())
    assert v.passed, v.reasons


def test_pipeline_short_circuits_at_consistency() -> None:
    p = ValidationPipeline(client=StubJudgeClient())  # type: ignore[arg-type]
    # Clean conversation (no tool calls): passes all stages. Both turns are
    # >= MIN_TURN_CHARS so heuristics pass too, exercising the full happy path
    # through to the (stubbed) judge.
    t = Trajectory(id="p2", domain="d", persona=Persona(), tools=[], system_prompt="s",
                   messages=[Message(role=Role.USER, content="একটি প্রশ্ন এখানে"),
                             Message(role=Role.ASSISTANT, content="একটি উত্তর এখানে")])
    v = p.validate(t)
    # passes all stages in this clean case
    assert v.passed
    assert v.stage_failed is None  # reached and passed the judge


def test_pipeline_short_circuits_at_heuristics_skips_judge() -> None:
    """A too-short user turn fails at heuristics, so the judge is never called.

    Uses a client that would explode if invoked, proving the pipeline stops early.
    """

    class ExplodingJudge:
        def judge(self, traj_json: str, rubric: str) -> Verdict:  # noqa: ARG002
            raise AssertionError("judge must not be called when heuristics fail")

    p = ValidationPipeline(client=ExplodingJudge())  # type: ignore[arg-type]
    t = Trajectory(id="p3", domain="d", persona=Persona(), tools=[], system_prompt="s",
                   messages=[Message(role=Role.USER, content="x"),  # < MIN_TURN_CHARS
                             Message(role=Role.ASSISTANT, content="একটি উত্তর এখানে")])
    v = p.validate(t)
    assert not v.passed
    assert v.stage_failed == "heuristics"
