"""SFT data for the small-model experiments: variants, dev carve-out, loss masking.

Pure Python (no torch) so the variant logic is unit-testable offline. The
tokenizer is passed in, only ``build_example`` touches it.
"""
from __future__ import annotations

import hashlib
import hashlib
import json
import random
from pathlib import Path

RELEASE = Path("outputs/release_multiturn/jontro_multiturn_2000_openai.jsonl")
TRAIN_IDS = Path("outputs/splits/train_ids.jsonl")

VARIANTS = ("full", "half", "bengali_only", "first_turn", "no_tui", "full_mix")
MIX = Path("mix/mix_v1.jsonl")


def load_train(release: Path = RELEASE, train_ids: Path = TRAIN_IDS) -> list[dict]:
    ids = {json.loads(line)["id"] for line in train_ids.open()}
    rows = [json.loads(line) for line in release.open()]
    return [r for r in rows if r["id"] in ids]


def goal_of(rec: dict) -> str:
    return rec["metadata"]["persona_goal"]


def split_dev(train: list[dict], frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Goal-disjoint (train, dev). Dev is for tuning only; test is never touched."""
    goals = sorted({goal_of(r) for r in train})
    rng = random.Random(seed)
    rng.shuffle(goals)
    dev_goals = set(goals[: max(1, round(len(goals) * frac))])
    return (
        [r for r in train if goal_of(r) not in dev_goals],
        [r for r in train if goal_of(r) in dev_goals],
    )


def truncate_after_first_call(rec: dict) -> dict:
    """Keep the prefix up to and including the first assistant tool-call message."""
    for i, m in enumerate(rec["messages"]):
        if m["role"] == "assistant" and m.get("tool_calls"):
            return {**rec, "messages": rec["messages"][: i + 1]}
    return rec


def load_mix(path: Path = MIX) -> list[dict]:
    """Self-distilled replay: base-model answers to unrelated prompts (see mix/ and the full_mix plan)."""
    return [json.loads(line) for line in path.open()]


def make_variant(train: list[dict], name: str, seed: int = 0, mix: list[dict] | None = None) -> list[dict]:
    if name == "full":
        return list(train)
    if name == "full_mix":  # Jontro train + self-distilled no-tool replay
        if not mix:
            raise ValueError("full_mix needs the mix records (mix/mix_v1.jsonl)")
        return list(train) + list(mix)
    if name == "half":  # data scale: half of the *goals*, not half of the records
        goals = sorted({goal_of(r) for r in train})
        random.Random(seed).shuffle(goals)
        keep = set(goals[: len(goals) // 2])
        return [r for r in train if goal_of(r) in keep]
    if name == "bengali_only":  # test on Banglish = cross-script transfer
        return [r for r in train if r["persona"]["script"] == "bengali"]
    if name == "no_tui":  # test on tui = unseen register
        return [r for r in train if r["persona"]["register"] != "tui"]
    if name == "first_turn":  # single-turn supervision: does multi-turn data matter?
        return [truncate_after_first_call(r) for r in train]
    raise ValueError(f"unknown variant {name!r}; choose from {VARIANTS}")


def to_template_messages(rec: dict) -> list[dict]:
    """Chat-template-ready messages: JSON-string arguments become dicts."""
    out = []
    for m in rec["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        if m["role"] == "assistant" and m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def _keep_hash(rec_id: str, i: int) -> float:
    """Deterministic uniform [0,1) for (record, message index)."""
    h = hashlib.sha256(f"{rec_id}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def _keep_hash(rec_id: str, i: int) -> float:
    """Deterministic uniform [0,1) for (record, message index)."""
    h = hashlib.sha256(f"{rec_id}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def build_examples(
    rec: dict, tok, max_len: int, text_turn_keep: float = 1.0
) -> tuple[list[dict], int]:
    """One training sample per assistant message, built exactly as inference builds it.

    Sample i is ``apply_chat_template(messages[:i], add_generation_prompt=True,
    enable_thinking=False)`` as the (masked) prompt and the assistant message i as the
    (labelled) completion. Templates such as Qwen3.5's render the empty think-block only on
    the final assistant turn, so a single-pass render of the whole trajectory is not
    prefix-consistent; per-turn samples sidestep that and match what vLLM sends at test time.
    Returns (samples, n_rejected). Over-long samples are rejected rather than truncated, since
    a truncated tool call would teach the model to stop mid-call.

    Cost control: every tool-call turn is kept (those are what the benchmark scores); plain-text
    turns are kept with probability ``text_turn_keep``, chosen deterministically per (record, turn).
    """
    msgs = to_template_messages(rec)
    tools = rec["tools"] or None  # mix "general chat" records offer no tools

    def ids(m: list[dict], gen: bool) -> list[int]:
        return tok.apply_chat_template(
            m, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False,
            enable_thinking=False,
        )

    out: list[dict] = []
    rejected = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        if not rec.get("keep_all_turns") and not m.get("tool_calls") and _keep_hash(rec["id"], i) >= text_turn_keep:
            continue
        prompt, full = ids(msgs[:i], True), ids(msgs[: i + 1], False)
        if full[: len(prompt)] != prompt or len(full) > max_len or len(full) == len(prompt):
            rejected += 1
            continue
        labels = [-100] * len(prompt) + full[len(prompt):]
        if tok.decode(full[-1:]) == "\n":  # template's trailing newline after <|im_end|>
            labels[-1] = -100
        if rec.get("truncated"):  # answer was cut at the generation cap: do not teach the model to stop there
            end = tok.convert_tokens_to_ids("<|im_end|>")
            last = max((j for j, t in enumerate(full) if t == end), default=None)
            if last is not None and last >= len(prompt):
                labels[last:] = [-100] * (len(labels) - last)
        out.append({"input_ids": full, "labels": labels})
    return out, rejected


def diagnose(rec: dict, tok) -> dict:
    """Why build_example rejects *rec*: lengths and the first non-append-only position."""
    msgs = to_template_messages(rec)
    tools = rec["tools"]

    def ids(m: list[dict], gen: bool) -> list[int]:
        return tok.apply_chat_template(
            m, tools=tools, add_generation_prompt=gen, tokenize=True, return_dict=False,
            enable_thinking=False,
        )

    full = ids(msgs, False)
    info: dict = {"full_len": len(full), "bad": []}
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        pre, post = ids(msgs[:i], True), ids(msgs[: i + 1], False)
        ok = full[: len(pre)] == pre and full[: len(post)] == post
        if not ok:
            k = next((j for j, (a, b) in enumerate(zip(full, post)) if a != b), min(len(full), len(post)))
            info["bad"].append({"msg": i, "pre_ok": full[: len(pre)] == pre,
                                "full_ctx": tok.decode(full[max(0, k - 30): k + 30]),
                                "post_ctx": tok.decode(post[max(0, k - 30): k + 30])})
            break
    return info
