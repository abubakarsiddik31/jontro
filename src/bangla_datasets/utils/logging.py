"""Structured run logging setup."""
import json
import logging
from datetime import UTC, datetime
from pathlib import Path


def configure_logging(run_dir: Path) -> Path:
    """Configure a run-scoped JSONL logger. Returns the log file path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log.jsonl"

    class JsonlHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "msg": record.getMessage(),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    root = logging.getLogger("bangla_datasets")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(JsonlHandler())
    return log_path
