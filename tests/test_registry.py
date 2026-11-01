import pytest

from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.utils.seeding import SeededRNG


def test_registry_get_known_tool() -> None:
    reg = ToolRegistry()
    t = reg.get("search_trains")
    assert t.name == "search_trains"


def test_registry_get_unknown_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_registry_subset_size() -> None:
    reg = ToolRegistry()
    rng = SeededRNG(seed=1)
    subset = reg.subset("railway_booking", 3, rng)
    assert len(subset) == 3


def test_registry_simulate_matches_simulator() -> None:
    reg = ToolRegistry()
    a = reg.simulate("search_trains", {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}, seed=7)
    b = reg.simulate("search_trains", {"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"}, seed=7)
    assert a == b
