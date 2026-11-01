"""Tests for the golden set loader + judge drift detection (Task 8)."""
from bangla_datasets.schema import Verdict
from bangla_datasets.validation.golden import DriftReport, GoldenSet


def test_golden_set_loads_18() -> None:
    gs = GoldenSet()
    trajs = gs.load()
    assert len(trajs) == 18
    labels = gs.labels()
    assert len(labels) == 18


def test_golden_set_ids_match_labels() -> None:
    gs = GoldenSet()
    ids = {t.id for t in gs.load()}
    labels = gs.labels()
    assert ids == set(labels.keys())


def test_golden_set_label_distribution() -> None:
    """11 pass=True, 7 pass=False — the hedge against judge drift."""
    labels = GoldenSet().labels()
    passes = sum(1 for v in labels.values() if v)
    fails = sum(1 for v in labels.values() if not v)
    assert passes == 11
    assert fails == 7


def test_check_drift_perfect_agreement() -> None:
    gs = GoldenSet()
    labels = gs.labels()
    verdicts = [Verdict(trajectory_id=tid, passed=expected) for tid, expected in labels.items()]
    report = gs.check_drift(verdicts)
    assert isinstance(report, DriftReport)
    assert report.agreement_rate == 1.0
    assert report.disagreements == []


def test_check_drift_below_threshold_flags_disagreements() -> None:
    gs = GoldenSet()
    labels = gs.labels()
    # Flip every label: 0% agreement, all disagree.
    verdicts = [Verdict(trajectory_id=tid, passed=not expected) for tid, expected in labels.items()]
    report = gs.check_drift(verdicts)
    assert report.agreement_rate == 0.0
    assert set(report.disagreements) == set(labels.keys())
