"""Turn the Modal mix candidates into SFT records: mix/mix_v1.jsonl (+ mix/MIX_STATS.md).

Keeps accepted candidates only, caps restraint at 600 and general at 300 (balanced across languages
by taking an even share from each before filling), and writes records in the same shape as Jontro
trajectories (`messages`, `tools`) with ``keep_all_turns`` so every assistant turn is trained on.

    uvx modal volume get jontro-out /mix out_modal/ --force
    python scripts/build_mix.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

CAP = {"restraint": 600, "general": 300}


def main() -> None:
    cand = [json.loads(line) for line in Path("out_modal/mix/candidates.jsonl").open()]
    rng = random.Random(0)
    kept: list[dict] = []
    for kind, cap in CAP.items():
        # accepted = clean, or cut at the generation cap (kept as `truncated`, trained without the end token)
        rows = [c for c in cand if c["kind"] == kind and c["reject_reason"] in (None, "unfinished")
                and not c["source"].startswith("synthetic")]  # synthetic Banglish dropped: the 2B rewriter's Romanisation is unusable
        rng.shuffle(rows)
        by_lang: dict[str, list[dict]] = {}
        for r in rows:
            by_lang.setdefault(r["lang"], []).append(r)
        chosen: list[dict] = []
        while len(chosen) < cap and any(by_lang.values()):  # round-robin across languages
            for lang in list(by_lang):
                if by_lang[lang] and len(chosen) < cap:
                    chosen.append(by_lang[lang].pop())
        kept += chosen
    out = []
    for i, c in enumerate(kept):
        msgs = ([{"role": "system", "content": c["system"]}] if c["system"] else []) + [
            {"role": "user", "content": c["prompt"]}, {"role": "assistant", "content": c["answer"]}]
        out.append({"id": f"mix-{c['kind'][0]}-{i:04d}", "kind": c["kind"], "lang": c["lang"], "source": c["source"],
                    "license": c["license"], "domain": c["domain"], "keep_all_turns": True, "truncated": c["reject_reason"] == "unfinished", "messages": msgs, "tools": c["tools"]})
    Path("mix/mix_v1.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    rej = Counter((c["kind"], c["lang"], c["reject_reason"]) for c in cand if c["reject_reason"] not in (None, "unfinished"))
    trunc = Counter((r["kind"], r["lang"]) for r in out if r["truncated"])
    kc = Counter((r["kind"], r["lang"]) for r in out)
    lines = ["# mix_v1 stats", "", f"records: {len(out)}", "", "| kind | lang | kept |", "|---|---|---:|"]
    lines += [f"| {k} | {l} | {n} |" for (k, l), n in sorted(kc.items())]
    lines += ["", f"truncated (cut at 512 tokens, end token not supervised): {sum(trunc.values())} of {len(out)}"]
    lines += ["", "| rejected | kind | lang | n |", "|---|---|---|---:|"] + [f"| {r} | {k} | {l} | {n} |" for (k, l, r), n in sorted(rej.items(), key=lambda x: str(x))]
    Path("mix/MIX_STATS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
