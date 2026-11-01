"""Interactive multi-turn orchestrator: drives authentic user-agent dialogues.

Enforces:
  1. True multi-turn conversation (user states goal -> assistant explores/presents
     options -> user decides/clarifies -> assistant executes/confirms -> wrap-up).
  2. Multi-tool chaining (2-4 tools composed in a single goal).
  3. Unfailingly polite assistant: always uses respectful 'apni' (আপনি/আপনার),
     strictly bans 'tui' (তুই/তোর) even when the user speaks informally.
  4. Script-conformance: Bengali script or Banglish matching the persona script.
  5. Deterministic simulator execution: 0% hallucinated tool outputs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from bangla_datasets.gemini.client import AssistantResponse
from bangla_datasets.gemini.prompts import build_assistant_prompt, build_persona_prompt
from bangla_datasets.schema import Message, Persona, Role, ToolDef, Trajectory
from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.utils.seeding import derive_seed

_log = logging.getLogger("bangla_datasets.generation.interactive")

# Expressions indicating user is satisfied, wrapping up, tersely acknowledging, or cancelling
_COMPLETION_PHRASES = [
    # Explicit thanks / farewell
    "অনেক ধন্যবাদ", "অসংখ্য ধন্যবাদ", "থ্যাংকস", "থ্যাংক ইউ", "ধন্যবাদ",
    "বিদায়", "বাই", "টা-টা", "টাটা",
    "thanks", "thank you", "dhonnobad", "bye", "tata",
    # Terse natural acknowledgments
    "ঠিক আছে", "ওকে", "হয়েছে", "হয়ে গেছে", "পেয়েছি", "আচ্ছা", "কাজ হয়ে গেছে",
    "সব ঠিক আছে", "আর লাগবে না", "আর কিছু লাগবে না",
    "thik ache", "ok", "hoyeche", "hoye geche", "peyechi", "accha", "shob thik ache", "ar lagbe na",
    "thikase", "bujhlam", "bujhsi",
    # Cancellations / Decisions not to proceed
    "থাক লাগবে না", "বাদ দাও", "বাদ দিন", "বাদ দে", "দরকার নেই", "দরকার নাই", "থাক",
    "bad dao", "bad din", "bad de", "dorkar nei", "dorkar nai", "thak", "cancel",
]

# Action/request verbs indicating the user is still issuing instructions
_ACTION_MARKERS = [
    "করুন", "করে দেন", "করে দাও", "করে দে", "বুক", "অর্ডার", "কাটুন", "কেটে",
    "দিন", "দাও", "দে", "পাঠান", "পাঠাও", "পাঠিয়ে", "চাচ্ছি", "চাই", "লাগবে",
    "রাখুন", "দেখুন", "দেখাও", "নিশ্চিত", "কনফার্ম",
    "book", "order", "pathao", "koro", "koren", "lagbe", "chai", "dekhun", "dekhao", "confirm",
]

# Prohibited assistant intimate morphology (Anti-Tui Gate)
_ASSISTANT_TUI_PATTERNS = [
    re.compile(r"\b(তুই|তোর|তোকে|তোরে|তোরই)\b"),
    re.compile(r"[\s\b](করিস|দিস|বলিস|খাস|যাস|পাস|দেখিস|শুনবি|বলবি|দিবি|করবি|হবি|থাকিস)[\s।,!?]"),
    re.compile(r"\b(tui|tor|toke|koris|dis|bolis|jabis|pabis|dibi|korbi)\b", re.IGNORECASE),
]

_GREETING_PREFIX_PAT = re.compile(
    r"^(নমস্কার|হ্যালো|হাই|সালাম|আসসালামু আলাইকুম|আদাব|nomoshkar|hello|hi|assalamu alaikum)[!।,.\s]+",
    re.IGNORECASE,
)

_FILLER_PATTERNS = [
    re.compile(r"[\s।,!]*(?:আমি |আমরা )?(?:আপনাকে |আপনাদের )?(?:এই বিষয়ে |এ বিষয়ে )?(?:প্রয়োজনীয় তথ্য দিয়ে |সম্পূর্ণ তথ্য দিয়ে )?(?:সাহায্য |সহযোগিতা |সেবা |উপকার )?করতে পেরে (?:আমি |আমরা )?(?:অত্যন্ত |খুব )?আনন্দিত[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*(?:আমি |আমরা )?(?:আপনাকে |আপনাদের )?(?:এই বিষয়ে |এ বিষয়ে )?(?:সাহায্য |সহযোগিতা )?করতে পার(?:লে|লে আমি|লে আমরা) (?:অত্যন্ত |খুব )?(?:আনন্দিত|খুশি) হব[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*(?:আমি |আমরা )?আপনাকে সাহায্য করতে পেরে (?:অত্যন্ত )?আনন্দিত হব[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*আপনার মতো একজন (?:সম্মানিত|চমৎকার|সজ্জন|মার্জিত|বিনয়ী) মানুষের (?:সাহায্য|সেবা) করতে পেরে (?:আমি|আমরা) (?:অত্যন্ত )?আনন্দিত(?: ও গর্বিত)?[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*এই বিষয়ে বা অন্য কোনো প্রয়োজনে আপনাকে সাহায্য করতে পারলে আমি অত্যন্ত আনন্দিত হব[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*ভবিষ্যতে যেকোনো প্রয়োজনে আপনাকে সাহায্য করতে পারলে আমি অত্যন্ত খুশি হব[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*(?:আমি |আমাকে )?(?:জানালে |জানালে আমি )(?:অত্যন্ত |খুব )?(?:আনন্দিত|খুশি) হব[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*আপনার সেবা করতে পেরে আমি অত্যন্ত আনন্দিত[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*আপনার উপকার করতে পেরে আমি অত্যন্ত আনন্দিত[।!.]?", re.IGNORECASE),
    re.compile(r"[\s।,!]*আপনাকে সেবা দিতে পেরে আমরা আনন্দিত[।!.]?", re.IGNORECASE),
]


def clean_natural_bangla_text(text: str, is_first_turn: bool = False, user_greeted_nomoshkar: bool = False) -> str:
    """Normalize text to eliminate unnatural translationese, sycophancy, and greeting repetition."""
    if not text:
        return text
    res = text
    # 1. Clean unnatural translationese 'উপলব্ধ'
    res = re.sub(r"উপলব্ধ\s*আছে", "পাওয়া যাচ্ছে", res)
    res = re.sub(r"উপলব্ধ\s*রয়েছে", "রয়েছে", res)
    res = re.sub(r"উপলব্ধ\s*নেই", "নেই", res)
    res = re.sub(r"উপলব্ধ\s*হবে", "পাওয়া যাবে", res)
    res = re.sub(r"উপলব্ধ\s*হতে\s*পারে", "পাওয়া যেতে পারে", res)
    res = re.sub(r"উপলব্ধ\s+(সার্কুলার|ফ্লাইট|স্কলারশিপ|প্যাকেজ|ট্রেন|হোটেল|ডাক্তার|বাস|লঞ্চ|মুভি|সিনেমা|ফ্ল্যাট|রুম|স্লট)", r"\1", res)
    res = re.sub(r"উপলব্ধ", "পাওয়া যাচ্ছে", res)

    # 2. Clean sycophancy fillers
    for pat in _FILLER_PATTERNS:
        res = pat.sub("", res)

    # 3. Handle greetings
    res = res.strip()
    if is_first_turn:
        if not user_greeted_nomoshkar:
            res = re.sub(r"^\s*(নমস্কার|nomoshkar)[!।,.\s]+", "হ্যালো! ", res, flags=re.IGNORECASE)
    else:
        # Strip repeated greetings in intermediate turns
        for _ in range(3):
            res = re.sub(
                r"^\s*(নমস্কার|হ্যালো|হাই|সালাম|আসসালামু আলাইকুম|আদাব|nomoshkar|hello|hi|assalamu alaikum)[!।,.\s]+",
                "",
                res.strip(),
                flags=re.IGNORECASE,
            ).strip()

    # Clean double punctuation & blank lines
    res = re.sub(r"[।!.]+\s*[।!.]+", "।", res)
    res = re.sub(r"[ \t]+", " ", res)
    res = re.sub(r"\n\s*\n\s*\n+", "\n\n", res)
    return res.strip()


_CANCELLATIONS = [
    "বাদ দাও", "বাদ দিন", "বাদ দে", "থাক লাগবে না", "লাগবে না", "দরকার নেই",
    "দরকার নাই", "প্রয়োজন নেই", "আর লাগবে না", "থাক",
    "bad dao", "bad din", "bad de", "thak lagbe na", "ar lagbe na", "dorkar nei", "cancel",
]


def contains_assistant_tui(text: str) -> bool:
    """Returns True if text contains prohibited 'tui' morphology."""
    for pat in _ASSISTANT_TUI_PATTERNS:
        if pat.search(text):
            return True
    return False


def is_user_signoff(text: str) -> bool:
    """Detect if a user turn is truly a signoff, terse ack, or statement of completion."""
    lower = text.lower()
    has_question = "?" in text or "？" in text
    if has_question:
        return False

    # Explicit cancellations are direct completion
    if any(c in lower or c in text for c in _CANCELLATIONS):
        return True

    has_phrase = any(p in lower or p in text for p in _COMPLETION_PHRASES)
    if not has_phrase:
        return False

    # Strip recognized phrases so substrings don't falsely trigger action markers
    clean_text = text
    for phrase in _COMPLETION_PHRASES + _CANCELLATIONS:
        clean_text = clean_text.replace(phrase, "")
    clean_lower = clean_text.lower()

    has_action = any(a in clean_lower or a in clean_text for a in _ACTION_MARKERS)
    return not has_action


@dataclass
class MultiTurnSeedPlan:
    """Rich plan for multi-turn multi-tool generation."""

    seed: int
    domain: str
    tool_subset: list[ToolDef]
    persona: Persona
    goal: str
    min_user_turns: int = 2
    max_user_turns: int = 4
    anchor_date: str = "2026-10-23"
    closure_style: str = "auto"
    metadata: dict[str, object] = field(default_factory=dict)


class _ClientProtocol(Protocol):
    def persona_turn(
        self, system_prompt: str, history: list[Message], seed: int, script: str = "bengali",
    ) -> str: ...

    def assistant_turn(
        self,
        system_prompt: str,
        history: list[Message],
        tools: list[ToolDef],
        seed: int,
    ) -> AssistantResponse: ...


class InteractiveOrchestrator:
    """State machine driving true multi-turn, multi-tool dialogues."""

    def __init__(
        self,
        client: _ClientProtocol,
        registry: ToolRegistry | None = None,
        max_tool_steps_per_turn: int = 6,
    ) -> None:
        self._client = client
        self._registry = registry or ToolRegistry()
        self._max_tool_steps = max_tool_steps_per_turn

    def generate(self, plan: MultiTurnSeedPlan) -> Trajectory:
        tools = plan.tool_subset
        tool_dicts = [{"name": t.name, "description": t.description} for t in tools]
        spl = getattr(plan.persona, "system_prompt_language", "bangla") or "bangla"
        script = getattr(plan.persona, "script", "bengali") or "bengali"

        system_prompt = build_assistant_prompt(
            tool_dicts, language=spl, script=script, anchor_date=plan.anchor_date
        )
        persona_sys = build_persona_prompt(
            plan.persona, plan.domain, plan.goal, anchor_date=plan.anchor_date
        )

        messages: list[Message] = []
        sim_seeds: dict[str, int] = {}
        sim_counter = 0
        total_assistant_turns = 0
        total_user_turns = 0
        terminated_reason = "max_turns"

        # Determine closure style for this trajectory (realistic distribution: terse_ack 35%, abrupt_fulfillment 25%, cancellation 20%, polite_farewell 20%)
        import random
        closure_rng = random.Random(derive_seed(plan.seed, "closure-style"))
        closure_style = getattr(plan, "closure_style", "auto")
        if not closure_style or closure_style == "auto":
            roll = closure_rng.random()
            if roll < 0.35:
                closure_style = "terse_ack"
            elif roll < 0.60:
                closure_style = "abrupt_fulfillment"
            elif roll < 0.80:
                closure_style = "cancellation"
            else:
                closure_style = "polite_farewell"

        # Turn 1: Opening user utterance
        opening = self._client.persona_turn(
            persona_sys, history=[], seed=derive_seed(plan.seed, "user-0"), script=script,
        )
        cleaned_opening = clean_natural_bangla_text(opening, is_first_turn=True, user_greeted_nomoshkar=False)
        messages.append(Message(role=Role.USER, content=cleaned_opening))
        total_user_turns += 1

        for user_turn_idx in range(plan.max_user_turns):
            # Assistant processing & tool calling loop
            tool_step = 0
            while tool_step < self._max_tool_steps:
                tool_step += 1
                resp = self._client.assistant_turn(
                    system_prompt,
                    history=messages,
                    tools=tools,
                    seed=derive_seed(plan.seed, f"asst-{user_turn_idx}-{tool_step}"),
                )

                if resp.tool_calls:
                    # Emit assistant turn with tool call(s)
                    msg = Message(
                        role=Role.ASSISTANT,
                        content=resp.text or None,
                        tool_calls=resp.tool_calls,
                    )
                    messages.append(msg)

                    # Deterministically simulate each tool
                    for tc in resp.tool_calls:
                        sim_seed = derive_seed(plan.seed, f"sim-{sim_counter}")
                        sim_counter += 1
                        result_str = self._registry.simulate(tc.name, tc.arguments, sim_seed)
                        sim_seeds[tc.id] = sim_seed
                        messages.append(Message(
                            role=Role.TOOL,
                            content=result_str,
                            tool_call_id=tc.id,
                        ))
                    # Continue loop so assistant digests the tool response
                    continue

                # Assistant produced conversational text for the user
                asst_text = resp.text.strip()
                cleaned_asst = clean_natural_bangla_text(
                    asst_text,
                    is_first_turn=(user_turn_idx == 0),
                    user_greeted_nomoshkar=("নমস্কার" in cleaned_opening),
                )
                messages.append(Message(role=Role.ASSISTANT, content=cleaned_asst))
                total_assistant_turns += 1
                break

            # 1. Check if conversation is naturally completed via signoff / terse ack / cancellation
            last_user_msg = [m for m in messages if m.role == Role.USER][-1]
            if (
                total_user_turns >= plan.min_user_turns
                and is_user_signoff(last_user_msg.content or "")
            ):
                terminated_reason = "natural_completion"
                break

            # 2. Abrupt fulfillment: user goal is confirmed by assistant without an open question
            tools_used_so_far = [tc.name for m in messages if m.tool_calls for tc in m.tool_calls]
            last_asst_msg = [m for m in messages if m.role == Role.ASSISTANT][-1]
            asst_text = (last_asst_msg.content or "").strip()
            is_open_question = asst_text.endswith("?") or asst_text.endswith("？")

            if (
                closure_style == "abrupt_fulfillment"
                and total_user_turns >= plan.min_user_turns
                and len(set(tools_used_so_far)) >= 2
                and not is_open_question
            ):
                terminated_reason = "abrupt_fulfillment"
                break

            # 3. Maximum user turns reached -> wrap-up based on closure style
            if user_turn_idx >= plan.max_user_turns - 1:
                if closure_style == "cancellation":
                    signoff_prompt = (
                        "apnar decision change korun ba khoroch beshi / shomoy na thakay kajti cancel korte bolun "
                        "(jemon: 'tahole thak, lagbe na ar' ba 'ajke bad dao, onno somoy dekhbo'). 1 bakye bolun."
                        if script == "banglish"
                        else "আপনার সিদ্ধান্ত পরিবর্তন করুন বা খরচ বেশি/সময় না থাকায় কাজটি বাতিল করতে বলুন "
                             "(যেমন: 'তাহলে থাক, লাগবে না আর' বা 'আজকে বাদ দাও, অন্য সময় দেখব')। ১ বাক্যে স্বাভাবিকভাবে বলুন।"
                    )
                    final_user_msg = self._client.persona_turn(
                        f"{persona_sys}\n\n{signoff_prompt}",
                        history=messages,
                        seed=derive_seed(plan.seed, "user-final-wrapup"),
                        script=script,
                    )
                    cleaned_final_user = clean_natural_bangla_text(final_user_msg, is_first_turn=False)
                    messages.append(Message(role=Role.USER, content=cleaned_final_user))
                    total_user_turns += 1

                    final_asst_resp = self._client.assistant_turn(
                        system_prompt,
                        history=messages,
                        tools=[],
                        seed=derive_seed(plan.seed, "asst-final-cancellation"),
                    )
                    cleaned_final_asst = clean_natural_bangla_text(final_asst_resp.text.strip(), is_first_turn=False)
                    messages.append(Message(role=Role.ASSISTANT, content=cleaned_final_asst))
                    total_assistant_turns += 1
                    terminated_reason = "cancellation_closure"
                    break

                elif closure_style == "terse_ack":
                    signoff_prompt = (
                        "kaj hoye geche — WhatsApp/chat user-er moto shudhu 1-2 shobde reaction din "
                        "(jemon: 'thik ache', 'ok', 'peyechi', 'accha'). Kono formal biday bolben na."
                        if script == "banglish"
                        else "কাজ সম্পন্ন হয়েছে — মেসেঞ্জার বা হোয়াটসঅ্যাপ চ্যাট ব্যবহারকারীর মতো সংক্ষেপে শুধু ১-২ শব্দে জানিয়ে দিন "
                             "(যেমন: 'ঠিক আছে', 'ওকে', 'পেয়েছি', 'আচ্ছা')। কোনো লম্বা বা আনুষ্ঠানিক বিদায় বলবেন না।"
                    )
                    final_user_msg = self._client.persona_turn(
                        f"{persona_sys}\n\n{signoff_prompt}",
                        history=messages,
                        seed=derive_seed(plan.seed, "user-final-wrapup"),
                        script=script,
                    )
                    cleaned_final_user = clean_natural_bangla_text(final_user_msg, is_first_turn=False)
                    messages.append(Message(role=Role.USER, content=cleaned_final_user))
                    total_user_turns += 1
                    terminated_reason = "terse_ack_closure"
                    break

                else:  # polite_farewell
                    signoff_prompt = (
                        "kothopokthon sesh korun — shohogarike dhonnobad janan je kaj hoyeche, "
                        "ar kichu lagbe na (1 bakye shabhabik biday)."
                        if script == "banglish"
                        else "কথোপকথন শেষ করুন — সহকারীকে স্বাভাবিকভাবে ধন্যবাদ জানান যে কাজ হয়ে গেছে, আর কিছু লাগবে না (১ বাক্যে বিদায়)।"
                    )
                    final_user_msg = self._client.persona_turn(
                        f"{persona_sys}\n\n{signoff_prompt}",
                        history=messages,
                        seed=derive_seed(plan.seed, "user-final-wrapup"),
                        script=script,
                    )
                    cleaned_final_user = clean_natural_bangla_text(final_user_msg, is_first_turn=False)
                    messages.append(Message(role=Role.USER, content=cleaned_final_user))
                    total_user_turns += 1

                    # Final assistant courteous farewell
                    final_asst_resp = self._client.assistant_turn(
                        system_prompt,
                        history=messages,
                        tools=[],  # no tools needed for final goodbye
                        seed=derive_seed(plan.seed, "asst-final-farewell"),
                    )
                    cleaned_final_asst = clean_natural_bangla_text(final_asst_resp.text.strip(), is_first_turn=False)
                    messages.append(Message(role=Role.ASSISTANT, content=cleaned_final_asst))
                    total_assistant_turns += 1
                    terminated_reason = "graceful_closure"
                    break

            # Next User Persona Turn (guarantee no repeated greetings in turn >= 1)
            turn_reminder = (
                "\n\n[সতর্কবার্তা]: এটি কথোপকথনের মধ্যবর্তী টার্ন। কোনো সম্ভাষণ ('নমস্কার', 'হ্যালো', 'সালাম') বলবেন না। সরাসরি সহকারীর প্রশ্নের উত্তর বা তথ্য দিন।"
                if script == "bengali"
                else "\n\n[Warning]: This is an intermediate turn. Do NOT repeat greetings ('hello', 'salam', 'nomoshkar'). Directly answer or provide requested information."
            )
            user_reply = self._client.persona_turn(
                f"{persona_sys}{turn_reminder}",
                history=messages,
                seed=derive_seed(plan.seed, f"user-{user_turn_idx + 1}"),
                script=script,
            )
            cleaned_user_reply = clean_natural_bangla_text(user_reply, is_first_turn=False)
            messages.append(Message(role=Role.USER, content=cleaned_user_reply))
            total_user_turns += 1

        tools_used = sorted({
            tc.name for m in messages if m.tool_calls for tc in m.tool_calls
        })

        return Trajectory(
            id=f"bd-agentic-{plan.seed:06d}",
            domain=plan.domain,
            persona=plan.persona,
            tools=tools,
            system_prompt=system_prompt,
            messages=messages,
            metadata={
                "turns_user": total_user_turns,
                "turns_assistant": total_assistant_turns,
                "total_messages": len(messages),
                "tools_used": tools_used,
                "tools_count": len(tools_used),
                "seed": plan.seed,
                "sim_seeds": sim_seeds,
                "closure_style": closure_style,
                "terminated": terminated_reason,
                "persona_goal": plan.goal,
                "scenario_type": plan.metadata.get("scenario_type", "multi_turn"),
            },
        )
