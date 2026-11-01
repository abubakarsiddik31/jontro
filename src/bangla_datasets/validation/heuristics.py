"""Stage 4: heuristic filters (pure Python).

Cheap pre-filters run before the expensive Gemini judge: turn-length sanity,
repeated identical tool calls, and script-mix detection.

The script check is *persona.script-aware*: a ``bengali`` trajectory must have
Bengali-script conversation turns; a ``banglish`` trajectory must have Latin
(Romanized) turns. A turn matching neither (e.g. English prose where Bangla or
Banglish is expected) is flagged as a language flip.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from bangla_datasets.generation.interactive_orchestrator import contains_assistant_tui
from bangla_datasets.schema import Role, Trajectory, Verdict
from bangla_datasets.utils.script import (
    cross_register_evidence,
    has_bengali,
    has_latin,
)

_SADHU_WORDS = [
    "মহাশয়", "মহাশয়", "মহাশয়া", "মহাশয়া", "উক্ত",
    # -ইতে infinitive family (Chalito uses -তে: জানতে, করতে, যেতে …)
    "জানিতে", "করিতে", "যাইতে", "দেখিতে", "বলিতে", "খাইতে", "লইতে",
    "শুনিতে", "পড়িতে", "পারিতে",
    # past/perfect -ইয়াছ- family (Chalito: -েছি / -েছিল)
    "করিয়াছি", "করিতেছে", "করিয়াছিল", "হইয়াছি", "ইয়াছি", "ইয়াছে",
    # future -ইব- family (Chalito: -বে / -ব)
    "যাইবে", "হইবেক", "হইবে", "করিবে", "দিবে", "জানাইবেন", "করিবেন",
    "যাইবেন", "দেখিবেন", "করিব", "যাইব", "হইব", "দেখিব", "শুনিব", "পড়িব",
    # oblique -ইয়া absolutive (Chalito: -ে / -িয়ে)
    "করিয়া", "হইয়া", "লইয়া", "গিয়া", "আসিয়া", "শুনিয়া", "দেখিয়া",
    # simple past -ইল (Chalito: -ল/-েছিল)
    "করিল", "হইল", "আসিল", "যাইল", "বলিল",
    # archaic pronouns/deictics
    "হইতে", "তাহারা", "তাহাদিগের", "যাহারা", "তাহার", "উপলব্ধ",
]
_SADHU_PATTERN = re.compile(r"(?:^|[\s।,!?])(" + "|".join(_SADHU_WORDS) + r")(?:$|[\s।,!?])")


def contains_sadhu_bhasha(text: str) -> bool:
    """Returns True if text contains archaic Sadhu Bhasha markers."""
    return bool(_SADHU_PATTERN.search(text))


MIN_TURN_CHARS = 3
# Turns shorter than this are not script-checked — too short to be meaningful.
_MIN_SCRIPT_CHECK_LEN = 20


def register_conformance(trajectory: Trajectory) -> dict[str, object]:
    """Structured register-conformance evidence for a trajectory's user turns.

    Splits the evidence into what is safe to gate on and what is not:

    - ``pronouns`` / ``imperatives`` — exact-token cross-register markers
      (তুই vs তুমি vs আপনি; দে vs দাও vs করুন). High precision; the gate fails
      on these.
    - ``verbs`` — suffix-matched cross-register verb morphology (``দিবি``,
      ``দেবে``, ``করবেন``). The original shipped gate discarded these entirely;
      they are reported as an advisory upper bound because short suffixes also
      terminate ordinary words.

    Only user turns longer than ``_MIN_SCRIPT_CHECK_LEN`` contribute.
    """
    declared = getattr(trajectory.persona, "register", "apni") or "apni"
    pronouns: list[str] = []
    imperatives: list[str] = []
    verbs: list[str] = []
    for m in trajectory.messages:
        if m.role != Role.USER or not m.content:
            continue
        if len(m.content) <= _MIN_SCRIPT_CHECK_LEN:
            continue
        ev = cross_register_evidence(m.content, declared)
        pronouns += [t for t in ev["pronouns"] if t not in pronouns]
        imperatives += [t for t in ev["imperatives"] if t not in imperatives]
        verbs += [t for t in ev["verbs"] if t not in verbs]
    return {
        "declared": declared,
        "pronouns": pronouns,
        "imperatives": imperatives,
        "verbs": verbs,
        "pronoun_conflict": bool(pronouns),
        "imperative_conflict": bool(imperatives),
        "verb_flag": bool(verbs),
    }


def run_heuristics(trajectory: Trajectory) -> Verdict:
    reasons: list[str] = []

    # length sanity
    for m in trajectory.messages:
        if (m.content is not None
                and len(m.content.strip()) < MIN_TURN_CHARS
                and m.role in (Role.USER, Role.ASSISTANT)):
            reasons.append(f"turn too short: {m.role.value}")

    # repeated identical tool calls (>=3 same name+args). Arguments are serialized
    # to a stable JSON string so list/dict-valued args (e.g. place_order's items
    # array) are hashable and comparable.
    calls = [
        (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
        for m in trajectory.messages if m.tool_calls for tc in m.tool_calls
    ]
    counts = Counter(calls)
    for call, n in counts.items():
        if n >= 3:
            reasons.append(f"repeated identical tool call x{n}: {call[0]}")

    # script-mix: enforce expected script on conversation turns.
    # User turns must match the persona's script (banglish -> Latin, bengali -> Bengali).
    # Assistant turns must ALWAYS be in standard contemporary Bengali script (বাংলা লিপি).
    script = getattr(trajectory.persona, "script", "bengali") or "bengali"
    for m in trajectory.messages:
        if m.role in (Role.USER, Role.ASSISTANT) and m.content:
            if len(m.content) <= _MIN_SCRIPT_CHECK_LEN:
                continue
            if m.role == Role.USER:
                if script == "banglish":
                    if not has_latin(m.content):
                        reasons.append(
                            "user turn has no Latin script (banglish expected)"
                        )
                else:
                    if not has_bengali(m.content):
                        reasons.append(
                            "user turn has no Bangla script (possible language flip)"
                        )
            elif m.role == Role.ASSISTANT:
                if not has_bengali(m.content):
                    reasons.append(
                        "assistant turn has no Bangla script (Bengali script expected)"
                    )

    # register-conformance: the persona's declared honorific level must match
    # the user turns. We gate on cross-register *pronouns* only — they are
    # unambiguous (তুই vs তুমি vs আপনি). Imperative stems (কর, বল) are shared
    # between tui and tumi, and verb endings overlap ordinary words, so both are
    # reported as advisory by register_conformance rather than gated here.
    conf = register_conformance(trajectory)
    if conf["pronouns"]:
        reasons.append(
            f"user turn uses cross-register pronoun(s) {conf['pronouns']} "
            f"but declared register is '{conf['declared']}'"
        )

    # anti-tui gate for assistant: assistant MUST NEVER speak in intimate 'tui'
    for m in trajectory.messages:
        if m.role == Role.ASSISTANT and m.content:
            if contains_assistant_tui(m.content):
                reasons.append("assistant turn contains prohibited 'tui' morphology")

    if reasons:
        return Verdict(trajectory_id=trajectory.id, passed=False, stage_failed="heuristics",
                       reasons=reasons)
    return Verdict(trajectory_id=trajectory.id, passed=True)


def validate_multi_turn_trajectory(
    trajectory: Trajectory,
    min_user_turns: int = 2,
    min_tools: int = 2,
    forbid_terminal_question: bool = True,
) -> Verdict:
    """Conference-grade validation for multi-turn, multi-tool trajectories."""
    base_verdict = run_heuristics(trajectory)
    reasons: list[str] = list(base_verdict.reasons)

    user_msgs = [m for m in trajectory.messages if m.role == Role.USER and m.content]
    asst_msgs = [m for m in trajectory.messages if m.role == Role.ASSISTANT and m.content]
    tool_calls = [
        tc.name for m in trajectory.messages if m.tool_calls for tc in m.tool_calls
    ]

    # 1. Multi-turn depth
    if len(user_msgs) < min_user_turns:
        reasons.append(f"insufficient user turns: {len(user_msgs)} < {min_user_turns}")
    if len(asst_msgs) < min_user_turns:
        reasons.append(f"insufficient assistant turns: {len(asst_msgs)} < {min_user_turns}")

    # 2. Multi-tool execution
    unique_tools = set(tool_calls)
    if len(unique_tools) < min_tools:
        reasons.append(f"insufficient distinct tools called: {len(unique_tools)} < {min_tools}")

    # 3. No dead-end terminal question
    if forbid_terminal_question and asst_msgs:
        last_asst_text = asst_msgs[-1].content.strip()
        if last_asst_text.endswith("?") or last_asst_text.endswith("？"):
            reasons.append("trajectory abruptly terminated on an unanswered assistant question")

    # 4. Register consistency is enforced by run_heuristics (cross-register
    #    pronouns and imperatives) and exposed structurally by
    #    register_conformance. Suffix verb morphology stays advisory, so it is
    #    deliberately NOT a gate reason here.

    # 5. Anti-Sadhu Bhasha check: strictly forbid archaic 19th-century prose
    for m in trajectory.messages:
        if m.content and contains_sadhu_bhasha(m.content):
            reasons.append(f"{m.role.value} turn contains archaic Sadhu Bhasha (modern Chalito expected)")

    if reasons:
        return Verdict(
            trajectory_id=trajectory.id,
            passed=False,
            stage_failed="multi_turn_quality_gate",
            reasons=reasons,
        )
    return Verdict(trajectory_id=trajectory.id, passed=True)

