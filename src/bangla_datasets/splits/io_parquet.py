"""Write/load dataset splits as Parquet (Hugging Face's recommended format).

Two formats per split:
- canonical: one row per trajectory, mirroring Trajectory.model_dump()
- sharegpt:  one row per trajectory with a "conversations" list of
             {from, value} dicts, reusing the existing to_sharegpt transform.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.parquet_codec import concrete_types, decode_row, encode_row
from bangla_datasets.writers.sharegpt import to_sharegpt


def write_split_parquet(
    trajectories: list[Trajectory],
    split_ids: set[str] | list[str],
    out_path: Path,
    fmt: str = "canonical",
) -> None:
    """Write the trajectories whose id is in split_ids to a Parquet file.

    Open-schema fields (tool-call ``arguments``, tool ``parameters_json_schema``)
    are stored as JSON strings so PyArrow's struct inference cannot pad them with
    nulls on the way back (see ``splits/parquet_codec``).
    """
    wanted = set(split_ids)
    rows: list[dict[str, Any]] = []
    for t in trajectories:
        if t.id not in wanted:
            continue
        if fmt == "canonical":
            rows.append(t.model_dump(mode="json"))
        elif fmt == "sharegpt":
            rows.append(to_sharegpt(t))
        else:
            raise ValueError(f"unknown fmt: {fmt!r}")
    rows = [encode_row(r) for r in rows]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = concrete_types(pa.Table.from_pandas(pd.DataFrame(rows)))
    pq.write_table(table, out_path)


def load_split_parquet(path: Path) -> list[dict[str, Any]]:
    """Load a Parquet split back to a list of row dicts (codec applied)."""
    table = pq.read_table(path)
    return [decode_row(r) for r in table.to_pylist()]
