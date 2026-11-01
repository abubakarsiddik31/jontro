"""Tests for Parquet split write/load round-trip."""
from __future__ import annotations

import json
from pathlib import Path

from bangla_datasets.schema import Trajectory
from bangla_datasets.splits.io_parquet import load_split_parquet, write_split_parquet

DATASET = Path("outputs/dataset/bangla_agentic.jsonl")


def _sample_trajectories(n: int = 50) -> list[Trajectory]:
    return [
        Trajectory.model_validate(json.loads(l))
        for l in DATASET.read_text().splitlines()[:n]
    ]


def test_roundtrip_canonical(tmp_path):
    trajs = _sample_trajectories(20)
    ids = [t.id for t in trajs]
    out = tmp_path / "train.parquet"
    write_split_parquet(trajs, ids, out, fmt="canonical")
    rows = load_split_parquet(out)
    assert len(rows) == 20
    assert {r["id"] for r in rows} == set(ids)
    # Nested fields survive the Parquet round-trip (messages is a list of structs).
    assert rows[0]["messages"][0]["role"] == trajs[0].messages[0].role.value


def test_roundtrip_sharegpt(tmp_path):
    trajs = _sample_trajectories(20)
    ids = [t.id for t in trajs]
    out = tmp_path / "train_sharegpt.parquet"
    write_split_parquet(trajs, ids, out, fmt="sharegpt")
    rows = load_split_parquet(out)
    assert len(rows) == 20
    # ShareGPT rows carry a "conversations" field (list of {from, value}).
    assert "conversations" in rows[0]


def test_subset_by_ids(tmp_path):
    """write_split_parquet only includes trajectories whose id is in split_ids."""
    trajs = _sample_trajectories(20)
    ids = [t.id for t in trajs[:5]]
    out = tmp_path / "subset.parquet"
    write_split_parquet(trajs, ids, out, fmt="canonical")
    rows = load_split_parquet(out)
    assert len(rows) == 5


def test_open_dict_fields_do_not_blow_up_the_schema(tmp_path):
    """sim_seeds is keyed by call id; it must be a JSON string, not one struct field per id. Empty columns need a real type."""
    import json as _json

    import pyarrow as _pa
    import pyarrow.parquet as _pq

    from bangla_datasets.splits.parquet_codec import concrete_types, decode_row, encode_row

    rows = [{"id": f"r{i}", "region": None, "metadata": {"sim_seeds": {f"call_{i}_{j}": j for j in range(3)}, "reasons": []}}
            for i in range(50)]
    enc = [encode_row(r) for r in rows]
    table = concrete_types(_pa.Table.from_pylist(enc))
    path = tmp_path / "t.parquet"
    _pq.write_table(table, path)
    back = _pq.read_table(path)
    assert not any(_pa.types.is_null(f.type) for f in back.schema)
    assert [f.name for f in back.schema.field("metadata").type] == ["sim_seeds", "reasons"]  # 2 fields, not 150
    assert [decode_row(r) for r in back.to_pylist()][7]["metadata"]["sim_seeds"] == rows[7]["metadata"]["sim_seeds"]
    assert isinstance(_json.loads(back.to_pylist()[0]["metadata"]["sim_seeds"]), dict)
