"""Hugging Face publish: parquet + dataset card + push."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.parquet_codec import encode_row
from bangla_datasets.writers.sharegpt import to_sharegpt

DATASET_CARD = """---
language:
  - bn
license: cc-by-4.0
task_categories:
  - text-generation
  - conversational
tags:
  - bangla
  - bengali
  - agentic
  - tool-use
  - function-calling
size_categories:
  - 1K<n<10K
---

# {repo_id}

Bangla multi-turn agentic tool-use trajectories. Conversation in Bangla (Bengali script);
tool/function schemas in English/code. Generated via a Gemini-driven self-play orchestrator
with deterministic tool simulators, validated through a 4-stage gate.

## Schema
Each row: `id`, `domain`, `tools` (English schemas), `conversations` (ShareGPT).

## License
CC-BY-4.0.
"""


def _to_dataframe(trajectories: list[Trajectory]) -> pd.DataFrame:
    rows = [encode_row(to_sharegpt(t)) for t in trajectories]
    return pd.DataFrame(rows)


def write_parquet(trajectories: list[Trajectory], out_dir: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _to_dataframe(trajectories)
    table = pa.Table.from_pandas(df)
    path = out_dir / "data.parquet"
    pq.write_table(table, path)
    return path


def publish_to_hf(
    trajectories: list[Trajectory],
    repo_id: str,
    token: str | None = None,
    out_dir: Path | str = "outputs/hf",
) -> str:
    """Write parquet + dataset card, then push to HF Hub. Returns the repo URL."""
    from huggingface_hub import HfApi

    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("No HF token. Set HF_TOKEN env var.")
    out_dir = Path(out_dir) / repo_id.replace("/", "__")
    write_parquet(trajectories, out_dir)
    (out_dir / "README.md").write_text(DATASET_CARD.format(repo_id=repo_id))
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
