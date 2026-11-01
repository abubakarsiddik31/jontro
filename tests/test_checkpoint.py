from pathlib import Path

from bangla_datasets.utils.checkpoint import Checkpoint


def test_checkpoint_records_completed(tmp_path: Path) -> None:
    cp = Checkpoint(run_dir=tmp_path)
    cp.mark_done("traj-001")
    assert cp.is_done("traj-001")
    assert not cp.is_done("traj-002")


def test_checkpoint_persists_across_instances(tmp_path: Path) -> None:
    cp1 = Checkpoint(run_dir=tmp_path)
    cp1.mark_done("traj-001")
    cp2 = Checkpoint(run_dir=tmp_path)
    assert cp2.is_done("traj-001")


def test_checkpoint_remaining(tmp_path: Path) -> None:
    cp = Checkpoint(run_dir=tmp_path)
    cp.mark_done("traj-001")
    assert cp.remaining(["traj-001", "traj-002", "traj-003"]) == ["traj-002", "traj-003"]


def test_checkpoint_handles_corrupted_file(tmp_path: Path) -> None:
    """A corrupted checkpoint.json (e.g. crash mid-write) must not kill resume:
    it should load as empty and treat nothing as done."""
    (tmp_path / "checkpoint.json").write_text("{corrupted")
    cp = Checkpoint(run_dir=tmp_path)  # should not crash
    assert cp.remaining(["a", "b"]) == ["a", "b"]
