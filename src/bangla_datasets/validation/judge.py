"""Stage 3: LLM judge — 5 scored dimensions via Gemini."""
from __future__ import annotations

import json

from bangla_datasets.gemini.prompts import build_judge_prompt
from bangla_datasets.schema import Trajectory, Verdict


def run_judge(trajectory: Trajectory, client: object, pass_threshold: int = 4) -> Verdict:
    """Score the trajectory. `client` must implement `.judge(traj_json, rubric) -> Verdict`.

    No network access happens here in tests — the client is a fake. The real
    GeminiClient (Task 4) implements the same `.judge` protocol.
    """
    traj_json = json.dumps(trajectory.model_dump(), ensure_ascii=False, indent=2)
    # Script-aware rubric: the bangla_fluency dimension is rephrased for Banglish
    # (Romanized) turns so the judge does not penalize legitimate banglish output.
    script = getattr(trajectory.persona, "script", "bengali") or "bengali"
    rubric = build_judge_prompt(script=script)
    verdict = client.judge(traj_json, rubric)  # type: ignore[attr-defined]
    # Enforce threshold + populate id.
    passed = all(s.score >= pass_threshold for s in verdict.scores) and len(verdict.scores) == 5
    return Verdict(
        trajectory_id=trajectory.id,
        passed=passed,
        stage_failed=None if passed else "judge",
        scores=verdict.scores,
    )
