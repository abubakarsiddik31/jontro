"""Generate the self-distilled replay for `full_mix`: the base model's OWN answers to unrelated prompts.

Component A (restraint): a Jontro system prompt + its tools are given, the user asks something unrelated;
the target is the base model's answer (kept only if it makes no tool call). Component B (general): no tools.
Banglish prompts are synthesised by the base model from Bengali prompts and language-checked. Candidates and
every rejection reason are written to /out/mix/candidates.jsonl so the filter is auditable.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from jontro_modal.retention import Generator

POOL = Path("/mix/prompts_v1.jsonl")
RELEASE = Path("/data/jontro_multiturn_2000_openai.jsonl")
TRAIN_IDS = Path("/data/train_ids.jsonl")
OUT = Path("/out/mix")
BN = re.compile(r"[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")
REWRITE = ("Rewrite the following Bangla sentence in Romanised Bangla (Banglish): Bangla words spelled with English letters "
           "only, the way people type on a phone. Output only the rewritten sentence, nothing else.\n\n{text}")
# allocations (candidates; kept counts are lower after filtering)
N = {"A_bn": 300, "B_bn": 130, "ban_src": 110, "A_ban": 60, "B_ban": 50, "A_en": 220, "B_en": 150}


def domain_contexts() -> list[tuple[str, str, list[dict]]]:
    """(domain, system_prompt, tools): first train trajectory of every domain."""
    train = {json.loads(line)["id"] for line in TRAIN_IDS.open()}
    seen: dict[str, tuple[str, list[dict]]] = {}
    for line in RELEASE.open():
        r = json.loads(line)
        if r["id"] in train and r["domain"] not in seen:
            seen[r["domain"]] = (r["messages"][0]["content"], r["tools"])
    return [(d, *seen[d]) for d in sorted(seen)]


def lang_ok(lang: str, prompt: str, out: str) -> bool:
    bn, la = len(BN.findall(out)), len(LATIN.findall(out))
    if bn + la == 0:
        return False
    if lang == "bn":
        return bn / (bn + la) >= 0.4
    if lang == "en":  # the Jontro system prompt is Bangla, so a Bengali reply to English is base behaviour; accept either
        return True
    return True


def run_mix(base_model: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(0)
    pool = [json.loads(line) for line in POOL.open()]
    bn = [r for r in pool if r["lang"] == "bn" and r["source"] != "authored"]
    en = [r for r in pool if r["lang"] == "en" and r["source"] != "authored"]
    chit = [r for r in pool if r["source"] == "authored"]
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    gen = Generator(model, tok)

    # 1) synthetic Banglish from held-out Bengali prompts
    src = bn[N["A_bn"] + N["B_bn"]: N["A_bn"] + N["B_bn"] + N["ban_src"]]
    rewrites = gen.generate([{"messages": [{"role": "user", "content": REWRITE.format(text=r["prompt"])}]} for r in src], 120, bs=32)
    ban = []
    for r, o in zip(src, rewrites):
        t = o["output"].strip().strip('"“”')
        if o["finished"] and "\n" not in t and 5 <= len(t) <= 220 and not BN.search(t) and len(LATIN.findall(t)) / max(len(t), 1) >= 0.6:
            ban.append({"lang": "banglish", "source": f"synthetic-from:{r['id']}", "license": r["license"], "prompt": t})
    A = (chit + bn[:N["A_bn"]] + en[:N["A_en"]] + ban[:N["A_ban"]])
    B = (bn[N["A_bn"]:N["A_bn"] + N["B_bn"]] + en[N["A_en"]:N["A_en"] + N["B_en"]] + ban[N["A_ban"]:N["A_ban"] + N["B_ban"]])
    ctxs = domain_contexts()
    rng.shuffle(A)
    items, meta = [], []
    for i, r in enumerate(A):  # A: restraint, Jontro system prompt + tools offered
        dom, sysp, tools = ctxs[i % len(ctxs)]
        items.append({"messages": [{"role": "system", "content": sysp}, {"role": "user", "content": r["prompt"]}], "tools": tools})
        meta.append({**r, "kind": "restraint", "domain": dom})
    for r in B:  # B: general chat, no system prompt, no tools
        items.append({"messages": [{"role": "user", "content": r["prompt"]}]})
        meta.append({**r, "kind": "general", "domain": None})
    order = sorted(range(len(items)), key=lambda i: len(items[i]["messages"][-1]["content"]))  # similar lengths per batch
    outs = [None] * len(items)
    for j, g in zip(order, gen.generate([items[i] for i in order], 512, bs=24)):
        outs[j] = g
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    with (OUT / "candidates.jsonl").open("w") as f:
        for m, it, g in zip(meta, items, outs):
            reason = None
            if not g["finished"]:
                reason = "unfinished"
            elif "<tool_call>" in g["output"]:
                reason = "tool_call"
            elif not g["output"].strip():
                reason = "empty"
            elif not lang_ok(m["lang"], m["prompt"], g["output"]):
                reason = "language"
            m = {**m, "answer": g["output"], "keep": reason is None, "reject_reason": reason,
                 "system": it["messages"][0]["content"] if it["messages"][0]["role"] == "system" else None,
                 "tools": it.get("tools") or []}
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            key = f"{m['kind']}/{m['lang']}/{'keep' if reason is None else reason}"
            stats[key] = stats.get(key, 0) + 1
    return {"synthetic_banglish": len(ban), "stats": dict(sorted(stats.items()))}
