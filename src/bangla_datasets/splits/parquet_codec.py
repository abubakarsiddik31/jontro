"""Faithful Parquet round-trip for open-schema fields.

``pyarrow`` infers a single struct type from the union of nested dict keys across
all rows, so a key present in one row and absent in another is written as
``null`` for the row that lacked it. Two dataset fields are genuinely
open-schema and therefore corrupted by that inference:

- ``tool_calls[].arguments`` — each call carries only its own parameters, so the
  union across 2,000 trajectories is the ~64-key merged parameter set. A 2-key
  ``search_properties`` call round-trips as 63 keys, most of them ``null``.
- ``tools[].parameters_json_schema`` — same effect on the advertised schema (a
  4-property tool becomes 71 properties).

- ``metadata.sim_seeds`` — a dict keyed by tool-call id, so Arrow made one struct field per id seen
  anywhere in the dataset (about 8,000 fields, a 112k-character schema), which ``datasets.load_dataset``
  cannot build. It is stored as a JSON string as well.

All three are stored as JSON **strings** instead (the OpenAI API already represents
arguments as a JSON string, so this is idiomatic), and parsed back on load. Every
other field keeps its native type.
"""
from __future__ import annotations

import contextlib
import copy
import json
from typing import Any

import pyarrow as pa

# Fields whose value is an open-schema dict that must survive exactly.
_OPEN_FIELDS = ("parameters_json_schema", "arguments", "sim_seeds")


def _stringify_open_fields(node: object) -> None:
    if isinstance(node, dict):
        for key, val in node.items():
            if key in _OPEN_FIELDS and isinstance(val, dict):
                node[key] = json.dumps(val, ensure_ascii=False, sort_keys=True)
            else:
                _stringify_open_fields(val)
    elif isinstance(node, list):
        for item in node:
            _stringify_open_fields(item)


def _parse_open_fields(node: object) -> None:
    if isinstance(node, dict):
        for key, val in node.items():
            if key in _OPEN_FIELDS and isinstance(val, str):
                with contextlib.suppress(json.JSONDecodeError):
                    node[key] = json.loads(val)
            else:
                _parse_open_fields(val)
    elif isinstance(node, list):
        for item in node:
            _parse_open_fields(item)


def encode_row(rec: dict[str, Any]) -> dict[str, Any]:
    """Copy *rec* with open-schema dict fields serialized to JSON strings."""
    out = copy.deepcopy(rec)
    _stringify_open_fields(out)
    return out


def decode_row(rec: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`encode_row`: parse the JSON-string fields back to dicts."""
    _parse_open_fields(rec)
    return rec


def _retype(t: pa.DataType) -> pa.DataType:
    if pa.types.is_null(t):
        return pa.string()
    if pa.types.is_struct(t):
        return pa.struct([pa.field(f.name, _retype(f.type)) for f in t])
    if pa.types.is_list(t):
        return pa.list_(_retype(t.value_type))
    return t


def concrete_types(table: pa.Table) -> pa.Table:
    """Give all-empty columns (inferred as ``null``) a real type, since some readers reject null-typed columns."""
    return table.cast(pa.schema([pa.field(f.name, _retype(f.type)) for f in table.schema]))
