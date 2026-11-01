"""Checkpoint manager for resumable generation runs."""
import json
from pathlib import Path


class Checkpoint:
    """Tracks which trajectory IDs have completed, persisted to disk."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.run_dir / "checkpoint.json"
        self._done: set[str] = set(self._load())

    def _load(self) -> list[str]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, ValueError):
                # A corrupted file (e.g. crash mid-write) must not kill resume.
                return []
            return data.get("done", []) if isinstance(data, dict) else []
        return []

    def _save(self) -> None:
        self._path.write_text(json.dumps({"done": sorted(self._done)}))

    def reconcile(self, jsonl_path: Path) -> None:
        """Prune done-entries whose trajectory is NOT in the output file.

        Makes resume crash-safe: if a process was killed after mark_done() but
        before the line was flushed (or the file was truncated by a fresh run),
        those IDs are re-queued for regeneration instead of being silently lost.
        """
        if not jsonl_path.exists():
            # File gone — everything must regenerate.
            if self._done:
                self._done.clear()
                self._save()
            return
        present: set[str] = set()
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "id" in obj:
                    present.add(obj["id"])
            except (json.JSONDecodeError, ValueError):
                continue
        stale = self._done - present
        if stale:
            self._done = self._done & present
            self._save()

    def mark_done(self, trajectory_id: str) -> None:
        self._done.add(trajectory_id)
        self._save()

    def is_done(self, trajectory_id: str) -> bool:
        return trajectory_id in self._done

    def remaining(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self._done]
