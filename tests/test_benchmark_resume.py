from pathlib import Path

from jontro_modal.benchmark import _append_partial, _load_partial


def test_partial_roundtrip_and_torn_line(tmp_path: Path):
    p = tmp_path / "x.partial.jsonl"
    _append_partial(p, [3, 1], [{"output": "a", "finished": True}, {"output": "b", "finished": False}])
    with p.open("a") as f:
        f.write('{"i": 9, "g": {"outp')  # killed mid-write
    done = _load_partial(p)
    assert done == {3: {"output": "a", "finished": True}, 1: {"output": "b", "finished": False}}
    assert _load_partial(tmp_path / "missing.jsonl") == {}
