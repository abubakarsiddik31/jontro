"""Validation pipeline: stages run cheap->expensive, short-circuit on first failure.

Execution order: schema (1) -> consistency (2) -> heuristics (4) -> judge (3).
Heuristics precede the judge because they are cheap pure-Python filters and the
judge is an expensive Gemini call.
"""
from __future__ import annotations

from bangla_datasets.schema import Trajectory, Verdict
from bangla_datasets.validation.consistency import check_consistency
from bangla_datasets.validation.heuristics import run_heuristics
from bangla_datasets.validation.judge import run_judge
from bangla_datasets.validation.schema_check import check_schema


class ValidationPipeline:
    """Runs the 4 stages in order. Short-circuits on first failure."""

    def __init__(self, client: object, pass_threshold: int = 4) -> None:
        self._client = client
        self._pass_threshold = pass_threshold

    def validate(self, trajectory: Trajectory) -> Verdict:
        # Stage 1
        v = check_schema(trajectory)
        if not v.passed:
            return v
        # Stage 2
        v = check_consistency(trajectory)
        if not v.passed:
            return v
        # Stage 4 (heuristics are cheap — run before the expensive judge)
        v = run_heuristics(trajectory)
        if not v.passed:
            return v
        # Stage 3 (expensive — Gemini)
        return run_judge(trajectory, self._client, pass_threshold=self._pass_threshold)

    def validate_batch(self, trajectories: list[Trajectory]) -> list[Verdict]:
        return [self.validate(t) for t in trajectories]
