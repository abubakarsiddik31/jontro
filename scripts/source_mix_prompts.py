"""Source and filter the prompt pools for the `full_mix` variant (no model calls, no cost).

Prompts only: the *answers* in the mix are the base model's own (self-distillation), so the only
licence question is the prompts. Sources are Apache-2.0 and human-written:
  CohereLabs/aya_dataset (Bengali)  and  OpenAssistant/oasst2 (English initial prompts).
Prompts are de-duplicated against the retention evaluation prompts so train and eval stay disjoint.

    python scripts/source_mix_prompts.py        # -> mix/prompts_v1.jsonl, mix/SOURCES.md
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset

MAX_CHARS = 220
SEED = 0
RET = [Path("retention/data/general_chat.jsonl"), Path("retention/data/tool_restraint.jsonl")]
BN = re.compile(r"[ঀ-৿]")

# Short chit-chat, authored here, disjoint from retention/data/tool_restraint.jsonl.
CHITCHAT = {
    "bn": ["শুভ সকাল!", "আজ কী খবর?", "আমার একটু একা লাগছে, কথা বলবে?", "তুমি কি আমার কথা বুঝতে পারো?", "ঠিক আছে, ধন্যবাদ তোমাকে।",
           "আচ্ছা, পরে আবার কথা হবে।", "তোমার সবচেয়ে প্রিয় ঋতু কোনটা?", "আজ খুব গরম পড়েছে, তাই না?", "একটা সুন্দর শুভরাত্রি বার্তা লেখো।", "তুমি কতটা বুদ্ধিমান?",
           "আমার জন্য একটা ছোট গল্প বলো।", "হাসির কিছু বলো তো।", "এক থেকে দশের মধ্যে একটা সংখ্যা ভাবো।", "চা না কফি, কোনটা ভালো?", "তোমার কি ঘুম পায়?"],
    "banglish": ["subho shokal!", "ki obostha tomar?", "ami ektu eka feel korchi, kotha bolbe?", "okay, ami bujhe gechi, onek dhonnobad.", "pore abar kotha hobe, bye.",
                 "tomar pochonder rong ki?", "aj onek gorom, na?", "ekta golpo shonao to.", "cha na coffee, kon ta bhalo?", "tumi ki ghumao?"],
    "en": ["Good morning!", "Hey, what's up?", "I'm a bit bored, can we chat?", "Do you understand me?", "Alright, thanks a lot.",
           "Talk to you later, bye.", "What's your favourite season?", "It's so hot today, isn't it?", "Write a nice goodnight message.", "Tell me a short story.",
           "Say something funny.", "Pick a number between one and ten.", "Tea or coffee?", "Do you ever get tired?", "Have a great day!"],
}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"[^\w]+", " ", t).strip()


def grams(t: str, n: int = 5) -> set[str]:
    t = norm(t).replace(" ", "")
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def too_close(p: str, ref_norms: set[str], ref_grams: list[set[str]]) -> bool:
    if norm(p) in ref_norms:
        return True
    g = grams(p)
    return any(len(g & r) / max(len(g | r), 1) > 0.6 for r in ref_grams)


def main() -> None:
    rng = random.Random(SEED)
    ref = [json.loads(line)["prompt"] for p in RET for line in p.open()]
    ref_norms, ref_grams = {norm(r) for r in ref}, [grams(r) for r in ref]

    def ok(p: str) -> bool:
        return (8 <= len(p) <= MAX_CHARS and "\n" not in p and "http" not in p and not too_close(p, ref_norms, ref_grams))

    aya = load_dataset("CohereLabs/aya_dataset", split="train")
    bn = sorted({r["inputs"].strip() for r in aya if r["language_code"] == "ben"})
    bn_ok = [p for p in bn if BN.search(p) and ok(p)]
    oasst = load_dataset("OpenAssistant/oasst2", split="train")
    en = sorted({r["text"].strip() for r in oasst if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en"})
    en_ok = [p for p in en if ok(p)]
    rng.shuffle(bn_ok)
    rng.shuffle(en_ok)

    rows = []
    for lang, plist, src, lic in (("bn", bn_ok, "CohereLabs/aya_dataset", "apache-2.0"), ("en", en_ok, "OpenAssistant/oasst2", "apache-2.0")):
        rows += [{"id": f"{lang}-{i:04d}", "lang": lang, "source": src, "license": lic, "prompt": p} for i, p in enumerate(plist)]
    for lang, ps in CHITCHAT.items():
        rows += [{"id": f"chit-{lang}-{i:02d}", "lang": lang, "source": "authored", "license": "cc-by-4.0", "prompt": p} for i, p in enumerate(ps)]
    chit_leak = [r["prompt"] for r in rows if r["source"] == "authored" and too_close(r["prompt"], ref_norms, ref_grams)]
    assert not chit_leak, chit_leak
    Path("mix").mkdir(exist_ok=True)
    Path("mix/prompts_v1.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    Path("mix/SOURCES.md").write_text(
        "# Prompt sources for full_mix\n\nOnly prompts are taken from these sources; answers are the base model's own.\n\n"
        "| Source | Licence | Language | Raw | Kept (<=220 chars, single line, no URL, not near an eval prompt) |\n|---|---|---|---:|---:|\n"
        f"| CohereLabs/aya_dataset | Apache-2.0 | Bengali | {len(bn)} | {len(bn_ok)} |\n"
        f"| OpenAssistant/oasst2 (initial prompts) | Apache-2.0 | English | {len(en)} | {len(en_ok)} |\n"
        f"| authored chit-chat | CC BY 4.0 | bn / Banglish / en | {sum(map(len, CHITCHAT.values()))} | {sum(map(len, CHITCHAT.values()))} |\n\n"
        "Excluded on licence: tatsu-lab/alpaca (CC BY-NC), BanglaLLM/bangla-alpaca-orca (GPL-3.0), databricks-dolly-15k (CC BY-SA, share-alike),\n"
        "and Bangla instruction sets with no declared licence.\n")
    print({"bn_raw": len(bn), "bn_kept": len(bn_ok), "en_raw": len(en), "en_kept": len(en_ok), "chit": sum(map(len, CHITCHAT.values()))})


if __name__ == "__main__":
    main()
