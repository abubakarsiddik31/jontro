"""Build the held-out human-text set for the retention NLL metric (prompt + human reply, Bengali and English).

Rows come from aya_dataset (Bengali prompt/answer pairs) and oasst2 (English prompt with its top-ranked assistant reply), both
Apache-2.0 and human-written. They are drawn from prompts that were NOT used in the full_mix training mix and are not retention eval prompts.

    python scripts/build_heldout_text.py   # -> retention/data/heldout_text.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

N = 100
MAX_PROMPT, MAX_REPLY = 400, 1200


def main() -> None:
    used = {json.loads(line)["prompt"] for line in Path("mix/mix_v1.jsonl").open() for _ in [0] if False}  # prompts live in messages
    mix_prompts = set()
    for line in Path("mix/mix_v1.jsonl").open():
        r = json.loads(line)
        mix_prompts |= {m["content"] for m in r["messages"] if m["role"] == "user"}
    pool_prompts = {json.loads(line)["prompt"] for line in Path("mix/prompts_v1.jsonl").open()}
    eval_prompts = {json.loads(line)["prompt"] for p in ("general_chat", "tool_restraint") for line in Path(f"retention/data/{p}.jsonl").open()}
    banned = mix_prompts | eval_prompts
    rng = random.Random(0)

    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    bn = [{"lang": "bn", "source": "CohereLabs/aya_dataset", "prompt": r["inputs"].strip(), "reply": r["targets"].strip()}
          for r in aya if r["language_code"] == "ben"]
    bn = [r for r in bn if r["prompt"] not in banned and 8 <= len(r["prompt"]) <= MAX_PROMPT and 40 <= len(r["reply"]) <= MAX_REPLY]

    oa = load_dataset("OpenAssistant/oasst2", split="train")
    by_parent: dict[str, list[dict]] = {}
    for r in oa:
        if r["role"] == "assistant" and r["parent_id"]:
            by_parent.setdefault(r["parent_id"], []).append(r)
    en = []
    for r in oa:
        if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en" and r["message_id"] in by_parent:
            best = min(by_parent[r["message_id"]], key=lambda x: (x["rank"] is None, x["rank"] or 0))
            en.append({"lang": "en", "source": "OpenAssistant/oasst2", "prompt": r["text"].strip(), "reply": best["text"].strip()})
    en = [r for r in en if r["prompt"] not in banned and 8 <= len(r["prompt"]) <= MAX_PROMPT and 40 <= len(r["reply"]) <= MAX_REPLY]
    rng.shuffle(bn)
    rng.shuffle(en)
    rows = [{"id": f"ho-{r['lang']}-{i:03d}", **r} for lang in ("bn", "en") for i, r in enumerate([x for x in (bn if lang == "bn" else en)][:N])]
    Path("retention/data/heldout_text.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print({"bn_available": len(bn), "en_available": len(en), "written": len(rows)})


if __name__ == "__main__":
    main()
