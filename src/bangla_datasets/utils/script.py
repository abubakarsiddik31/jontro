"""Centralized Bangla/Banglish script detection.

Previously the Bangla-script range regex ``[\\u0980-\\u09FF]`` was duplicated
across ``schema.py``, ``validation/consistency.py``, and ``validation/heuristics.py``.
This module is the single source of truth so the three sites cannot drift.
"""
from __future__ import annotations

import re

# Bengali Unicode block: U+0980–U+09FF.
BANGLA_RANGE_RE = re.compile(r"[\u0980-\u09FF]")

# An ASCII identifier: ``[A-Za-z_][A-Za-z0-9_]*``. The English/code layer.
_ASCII_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Latin letters A-Z, a-z — used to recognize Banglish (Romanized Bangla).
_LATIN_RE = re.compile(r"[A-Za-z]")


def has_bengali(text: str) -> bool:
    """True if *text* contains at least one Bengali-script character."""
    return bool(BANGLA_RANGE_RE.search(text))


def has_latin(text: str) -> bool:
    """True if *text* contains at least one Latin letter."""
    return bool(_LATIN_RE.search(text))


def is_banglish(text: str) -> bool:
    """Heuristic: Romanized Bangla — has Latin letters but no Bengali script.

    Real Banglish turns (``ami dhaka jete chai``) are Latin-only; genuine Bangla
    turns (``আমি ঢাকা যেতে চাই``) are Bengali-script. This distinguishes them so the
    script-aware heuristics know which script requirement to enforce.
    """
    return has_latin(text) and not has_bengali(text)


def is_ascii_identifier(name: str) -> bool:
    """True if *name* is a valid ASCII identifier (English/code layer)."""
    return bool(_ASCII_IDENT_RE.match(name))


def has_bangla_key(key: str) -> bool:
    """True if a string key contains any Bengali character (tool-layer violation)."""
    return bool(BANGLA_RANGE_RE.search(key))


# ---------------------------------------------------------------------------
# Speech-register (honorific level) marker detection.
#
# Bangla has three honorific levels encoded in pronouns + verb morphology:
#   tui  (intimate)  — তুই/তোর, করিস/দিস/দে/বলিস/রে
#   tumi (familiar)  — তুমি/তোমাকে, করো/দেখো/দাও/বলো/দেবে
#   apni (formal)    — আপনি, করবেন/পারবেন/করুন/অনুগ্রহ/দেবেন
#
# We tokenize on whitespace/Bengali punctuation (NOT \b — Python's \b uses
# Unicode word-boundary rules that misfire on Bengali combining marks, the
# category-Mn vowel signs that attach to consonants). Matching whole tokens
# avoids the substring false positives that plague naive `in` checks.
# ---------------------------------------------------------------------------

# Token-terminal markers: each register has characteristic verb endings and
# particles. A token "ends with" one of these iff token == suffix OR
# token[-len(suffix):] == suffix (so "করবেন" matches the "বেন" suffix but
# "শীর্ষে" does not falsely match "রে"). We require the token to be short
# enough that the suffix IS the morpheme, not a coincidence in a long word.

# Pronouns — exact-token matches (strongest signal).
_PRONOUNS = {
    "tui":  {"তুই", "তোর", "তোকে", "তোরা", "tui", "tor", "toke"},
    "tumi": {"তুমি", "তোমার", "তোমাকে", "তোমরা", "tumi", "tomar", "tomake"},
    "apni": {"আপনি", "আপনার", "আপনাকে", "আপনারা",
             "মহাশয়", "মহাশয়", "মহাশয়া", "মহাশয়া",
             "apni", "apnar", "mohashoy", "mohashoya"},
}

# Verb-ending suffixes — matched as token suffixes (token ends with these).
# Kept to unambiguous morphemes only; bare single-letter suffixes (ব, ও, ন)
# are too noisy (থাকব, যাও, তিন) so we rely on the longer 2-char forms and the
# exact-token set above for those registers.
_VERB_SUFFIXES = {
    "tui":  ["বি", "বিস", "ইস", "ছিস", "িস"],   # দিবি, করবিস, করিস, করছিস, বলিস
    "tumi": ["বে"],                              # করবে, দেবে, যাবে
    "apni": ["বেন", "রুন", "ুন"],                # করবেন, করুন, করুন
}

# A small set of high-precision whole-token markers (fillers, imperatives).
# NOTE: Bengali has TWO Unicode forms for the "o" sound — the vowel SIGN ো
# (U+09CB, attaches to a consonant: খো) and the independent vowel ও (U+0993,
# stands alone: খাও). Imperatives use both across writers, so we list both.
_EXACT_TOKENS = {
    "tui":  {"দে", "রে", "আয়", "আয়", "যা", "শোন", "শন"},
    "tumi": {
        # vowel-sign ো forms (দেখো, করো, বলো, শোনো, নাও, পাঠাও)
        "দাও", "করো", "দেখো", "বলো", "পাঠাও", "শোনো", "বল", "কর", "নাও", "যাও",
        # independent-ও imperative forms (দেখাও, করাও, বলাও, নাও, পাঠাও)
        "দেখাও", "করাও", "বলাও", "খাও", "যাও", "পাঠাও",
    },
    "apni": {
        "অনুগ্রহ", "করুন", "জানাবেন", "দেখান", "বলবেন", "পারবেন", "করবেন", "দেবেন",
        "নমস্কার", "দয়া", "দয়া", "শ্রদ্ধেয়", "শ্রদ্ধেয়", "আগ্রহ",
        "হবেন", "করতেন", "পারতেন", "দিতেন",
    },
}

# Latin-script (Banglish) exact-token markers.
_LATIN_TOKENS = {
    "tui":  {"de", "re", "tui", "tor", "toke", "bolis", "koris", "dis", "shon", "ja"},
    "tumi": {"dao", "koro", "dekho", "dekhaow", "bolo", "pathao", "shono", "tumi", "tomar",
             "bal", "kor", "jao", "khab"},
    "apni": {"onugroho", "korun", "korben", "deben", "parben", "janaben", "apni", "apnar",
             "mohashoy", "nomoskar", "daya", "hoben"},
}


def _tokens(text: str) -> list[str]:
    """Split on whitespace and common Bengali/Persian punctuation.

    Avoids \\b (unreliable on Bengali combining marks). Returns lowercased
    Latin tokens and original-case Bengali tokens.
    """
    # Replace Bengali danda, persian period, and punctuation with space.
    cleaned = re.sub(r"[।\?\!\,\.\;\:\"\'\(\)\[\]\{\}\-—–…০-৯]", " ", text)
    return [t for t in cleaned.split() if t]


def _ends_with(token: str, suffix: str) -> bool:
    """True if *token* ends with *suffix* AND token is short enough that the
    suffix is the morpheme, not a coincidental tail of a long unrelated word.

    For a 2-char suffix we require token length <= 6; for >=3 chars, <= 8.
    This lets 'দিবি' match suffix 'বি' but blocks 'শ্রমিকজীবি' (8 chars, a noun).
    """
    if not token.endswith(suffix):
        return False
    max_len = 6 if len(suffix) <= 2 else 8
    return len(token) <= max_len


def register_markers(text: str) -> dict[str, list[str]]:
    """Return the markers of each speech register found in *text*.

    Returns ``{"tui": [...], "tumi": [...], "apni": [...]}`` where each list
    holds the matched tokens. Empty lists mean no markers of that register.
    Used by the register-conformance heuristic and by manual audits.
    """
    toks = _tokens(text)
    found: dict[str, list[str]] = {"tui": [], "tumi": [], "apni": []}

    def _add(reg: str, tok: str) -> None:
        key = tok.lower() if tok.isascii() else tok
        if key not in found[reg]:
            found[reg].append(key)

    for t in toks:
        low = t.lower()
        for reg in ("tui", "tumi", "apni"):
            # exact-token match (pronouns, imperatives, latin markers)
            if t in _PRONOUNS[reg] or t in _EXACT_TOKENS[reg]:
                _add(reg, t)
            elif low in _LATIN_TOKENS[reg]:
                _add(reg, low)
            else:
                # verb-suffix match (Bengali morphology)
                for suf in _VERB_SUFFIXES[reg]:
                    if _ends_with(t, suf):
                        _add(reg, t)
                        break
    return found


def dominant_register(text: str) -> str | None:
    """Return the dominant speech register of *text*, or None if undecided.

    Counts marker tokens per register; returns the one with the most. Ties and
    zero-marker texts return None (caller decides the fallback).
    """
    found = register_markers(text)
    counts = {reg: len(ms) for reg, ms in found.items()}
    top = max(counts.values())
    if top == 0:
        return None
    winners = [reg for reg, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else None


def cross_register_evidence(text: str, declared: str) -> dict[str, list[str]]:
    """Markers in *text* that belong to a *different* register than ``declared``.

    Returns ``{"pronouns", "imperatives", "verbs"}``:

    - ``pronouns`` (তুই / তুমি / আপনি …) and ``imperatives`` (দে / দাও / করুন …)
      are exact-token matches, so they are high-precision and safe to gate on.
    - ``verbs`` are suffix matches and are deliberately reported separately as
      *advisory*: short Bengali verb endings also terminate ordinary words (the
      familiar suffix ``বে`` ends the conjunction ``তবে``), so a cross-register
      verb hit bounds the mislabelling rate rather than proving an error.

    Unknown registers yield an empty result rather than raising.
    """
    out: dict[str, list[str]] = {"pronouns": [], "imperatives": [], "verbs": []}

    def _add(kind: str, tok: str) -> None:
        key = tok.lower() if tok.isascii() else tok
        if key not in out[kind]:
            out[kind].append(key)

    for t in _tokens(text):
        low = t.lower()
        for reg in ("tui", "tumi", "apni"):
            if reg == declared:
                continue
            if t in _PRONOUNS[reg] or (t.isascii() and low in _PRONOUNS[reg]):
                _add("pronouns", t)
            elif t in _EXACT_TOKENS[reg] or (t.isascii() and low in _LATIN_TOKENS[reg]):
                _add("imperatives", t)
            else:
                for suf in _VERB_SUFFIXES[reg]:
                    if _ends_with(t, suf):
                        _add("verbs", t)
                        break
    return out
