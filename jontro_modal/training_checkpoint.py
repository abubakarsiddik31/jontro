"""Atomic, immutable checkpoints for the Modal SFT trainer.

Each optimizer step is written to its own directory.  ``checkpoint.json`` is
only a small atomic pointer to the latest completed checkpoint; it is never a
checkpoint payload itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1
POINTER_NAME = "checkpoint.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    model: Any,
    opt: Any,
    sched: Any,
    losses: list[Any],
    step: int,
    total_steps: int,
    identity: dict[str, Any],
    root: str | os.PathLike[str],
) -> Path:
    """Save an immutable checkpoint and atomically point ``root`` to it.

    If writing fails, the previous pointer remains intact and the staging
    directory is removed. A step directory is never overwritten.
    """
    if step < 0 or total_steps < 0 or step > total_steps:
        raise ValueError("step must be between zero and total_steps")
    if not isinstance(identity, dict):
        raise TypeError("identity must be a JSON-serializable dict")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    final_dir = root_path / f"step_{step}"
    if final_dir.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {final_dir}")

    staging = Path(tempfile.mkdtemp(prefix=f".step_{step}.", dir=root_path))
    try:
        adapter_dir = staging / "adapter"
        model.save_pretrained(adapter_dir)
        state = {
            "step": step,
            "total_steps": total_steps,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "losses": losses,
            "cpu_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
        }
        torch.save(state, staging / "state.pt")
        meta = {
            "format_version": FORMAT_VERSION,
            "done": True,
            "step": step,
            "total_steps": total_steps,
            "identity": identity,
            "files": _files_hashes(staging),
        }
        with (staging / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(meta, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-filesystem directory rename publishes the complete checkpoint.
        os.rename(staging, final_dir)
        _atomic_json(root_path / POINTER_NAME, {
            "format_version": FORMAT_VERSION,
            "done": True,
            "step": step,
            "total_steps": total_steps,
            "path": final_dir.name,
        })
        return final_dir
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def load_checkpoint(
    root: str | os.PathLike[str],
    identity: dict[str, Any],
    total_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    """Validate and load the latest checkpoint, or return ``None`` if absent.

    Invalid, stale, tampered, or legacy root-level markers raise ``ValueError``.
    Return ``(metadata, state, adapter_model.safetensors_path)`` so the caller
    can restore the adapter without guessing the saved layout.
    """
    root_path = Path(root)
    pointer_path = root_path / POINTER_NAME
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("invalid checkpoint pointer JSON") from exc
    if pointer.get("format_version") != FORMAT_VERSION or "path" not in pointer:
        raise ValueError("legacy or unsupported checkpoint marker; refusing resume")
    if pointer.get("done") is not True:
        raise ValueError("checkpoint pointer is not marked complete")
    if pointer.get("total_steps") != total_steps:
        raise ValueError("checkpoint total_steps mismatch")
    rel = Path(pointer["path"])
    if rel.is_absolute() or len(rel.parts) != 1 or rel.name != f"step_{pointer.get('step')}":
        raise ValueError("invalid checkpoint path in pointer")
    checkpoint_dir = root_path / rel
    try:
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(
            encoding="utf-8"))
    except Exception as exc:
        raise ValueError("checkpoint metadata is missing or invalid") from exc
    if metadata.get("format_version") != FORMAT_VERSION or metadata.get("done") is not True:
        raise ValueError("checkpoint metadata is incomplete or unsupported")
    if metadata.get("identity") != identity:
        raise ValueError("checkpoint identity mismatch")
    if metadata.get("step") != pointer.get("step"):
        raise ValueError("checkpoint step mismatch")
    if metadata.get("total_steps") != total_steps:
        raise ValueError("checkpoint metadata total_steps mismatch")
    expected = metadata.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("checkpoint file hashes missing")
    actual = _files_hashes(checkpoint_dir)
    if actual != expected:
        raise ValueError("checkpoint file hash mismatch")
    state_path = checkpoint_dir / "state.pt"
    adapter_path = checkpoint_dir / "adapter" / "adapter_model.safetensors"
    if "state.pt" not in expected or not adapter_path.is_file():
        raise ValueError("checkpoint payload is incomplete")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("step") != pointer["step"] or state.get("total_steps") != total_steps:
        raise ValueError("checkpoint state metadata mismatch")
    return metadata, state, str(adapter_path)
