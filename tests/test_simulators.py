import pytest

from bangla_datasets.tools.simulators import simulate


def test_simulator_deterministic_same_seed() -> None:
    a = simulate("search_trains", {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}, seed=42)
    b = simulate("search_trains", {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}, seed=42)
    assert a == b


def test_simulator_returns_json_str() -> None:
    out = simulate("search_trains", {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}, seed=1)
    assert isinstance(out, str)
    assert "ঢাকা" in out or "চট্টগ্রাম" in out or "train" in out.lower()


def test_simulator_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        simulate("does_not_exist", {}, seed=1)
