import json

import pytest

from jontro_modal.sft_data import (
    VARIANTS,
    goal_of,
    load_train,
    make_variant,
    split_dev,
    to_template_messages,
    truncate_after_first_call,
)


@pytest.fixture(scope="module")
def train():
    return load_train()


def test_train_size(train):
    assert len(train) == 1300


def test_dev_split_is_goal_disjoint(train):
    tr, dev = split_dev(train)
    assert dev and tr
    assert not {goal_of(r) for r in tr} & {goal_of(r) for r in dev}
    assert len(tr) + len(dev) == len(train)


def test_variants_semantics(train):
    assert len(make_variant(train, "full")) == 1300
    half = make_variant(train, "half")
    assert 0.3 < len(half) / 1300 < 0.7
    assert {r["persona"]["script"] for r in make_variant(train, "bengali_only")} == {"bengali"}
    assert "tui" not in {r["persona"]["register"] for r in make_variant(train, "no_tui")}
    for r in make_variant(train, "first_turn"):
        calls = [m for m in r["messages"] if m.get("tool_calls")]
        assert len(calls) == 1 and r["messages"][-1] is calls[0]


def test_all_variants_are_train_only(train):
    ids = {r["id"] for r in train}
    for v in VARIANTS:
        if v == "full_mix":
            continue  # needs mix records; covered below
        assert {r["id"] for r in make_variant(train, v)} <= ids


def test_arguments_become_dicts(train):
    for m in to_template_messages(train[0]):
        for tc in m.get("tool_calls") or []:
            assert isinstance(tc["function"]["arguments"], dict)
        if m["role"] == "assistant":
            assert m["content"] is not None


def test_truncate_noop_without_calls():
    rec = {"messages": [{"role": "user", "content": "x"}]}
    assert truncate_after_first_call(rec) is rec


def test_full_mix_requires_and_appends_mix(train):
    mix = [{"id": "mix-r-0000", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "tools": [], "keep_all_turns": True}]
    out = make_variant(train, "full_mix", mix=mix)
    assert len(out) == len(train) + 1 and out[-1]["id"] == "mix-r-0000"
    with pytest.raises(ValueError):
        make_variant(train, "full_mix")
