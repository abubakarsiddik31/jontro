"""Typed Gemini client: persona/user, assistant, judge. Retry + cost-capped."""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from bangla_datasets.schema import JudgeScore, Message, Role, ToolCall, ToolDef, Verdict

_log = logging.getLogger("bangla_datasets.gemini")

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_RETRIES = 3

# Target-register descriptors for user-turn rewrite. Keyed by (register, script).
_REWRITE_TARGETS: dict[tuple[str, str], str] = {
    ("tui", "bengali"): (
        "tui (intimate): তুই/তোর সম্বোধন। খুব কাছের বন্ধুর মতো আলগা ভাষা। "
        "ক্রিয়াপদ: করিস, দিস, দে, বলিস, যা, বল, দেখিস। সর্বনাম: তুই, তোর, তোকে।"
    ),
    ("tui", "banglish"): (
        "tui (intimate): tui/tor shombodhon. khub kacher bondhur moto alga bhasha. "
        "krijapod: koris, dis, de, bolis, ja, bol, dekhis. sorbonam: tui, tor, toke."
    ),
    ("tumi", "bengali"): (
        "tumi (familiar): তুমি/তোমাকে সম্বোধন। বন্ধু বা পরিচিতের মতো বন্ধুত্বপূর্ণ ভাষা। "
        "ক্রিয়াপদ: করো, দেখো, দাও, বলো, কর, দেবে। সর্বনাম: তুমি, তোমার, তোমাকে।"
    ),
    ("tumi", "banglish"): (
        "tumi (familiar): tumi/tomake shombodhon. bondhu ba porichiter moto "
        "bondhuttopurno bhasha. krijapod: koro, dekho, dao, bolo, kor, debe. "
        "sorbonam: tumi, tomar, tomake."
    ),
    ("apni", "bengali"): (
        "apni (formal): আপনি সম্বোধন। শ্রদ্ধাশীল ও আনুষ্ঠানিক ভাষা। "
        "ক্রিয়াপদ: করবেন, পারবেন, করুন, দেবেন, জানাবেন। সর্বনাম: আপনি, আপনার।"
    ),
    ("apni", "banglish"): (
        "apni (formal): apni shombodhon. shroddhashil o anusthanik bhasha. "
        "krijapod: korben, parben, korun, deben, janaben. sorbonam: apni, apnar."
    ),
}


@dataclass
class AssistantResponse:
    """Parsed output of an assistant turn."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class GeminiClient:
    """Wraps google-genai with typed methods + retry + cost tracking."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_input_tokens_per_run: int | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY env var."
            )
        self._api_key = key
        self._model = model
        self._max_retries = max_retries
        self._max_input_tokens = max_input_tokens_per_run
        self._tokens_used = 0
        self._client = genai.Client(api_key=key)

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    def _with_retry(
        self,
        fn: Callable[[], types.GenerateContentResponse],
    ) -> types.GenerateContentResponse:
        """Retry with exponential backoff on transient errors."""
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last_err = e
                _log.warning("Gemini call failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {self._max_retries} retries") from last_err

    def _check_budget(self, est_tokens: int) -> None:
        if (
            self._max_input_tokens is not None
            and self._tokens_used + est_tokens > self._max_input_tokens
        ):
            raise RuntimeError(
                f"Cost cap exceeded: {self._tokens_used + est_tokens} > {self._max_input_tokens}"
            )

    def _track_usage(self, response: types.GenerateContentResponse, fallback_text: str) -> None:
        """Count input tokens from real usage_metadata when present, else estimate.

        Real usage is the authoritative signal returned by the API; the
        len//4 + constant estimate is a conservative fallback only.
        """
        usage = getattr(response, "usage_metadata", None)
        if usage is not None and getattr(usage, "prompt_token_count", None):
            self._tokens_used += usage.prompt_token_count or 0
        else:
            self._tokens_used += len(fallback_text) // 4 + 200

    def persona_turn(
        self, system_prompt: str, history: list[Message], seed: int, script: str = "bengali",
    ) -> str:
        """Ask Gemini-as-user for the next user turn.

        ``script`` selects the opening-seed language so the persona's first
        utterance matches its conversation-script axis (bengali | banglish).
        """
        self._check_budget(est_tokens=800)
        if history:
            contents = "\n".join(f"{m.role.value}: {m.content or ''}" for m in history if m.content)
        else:
            # Opening turn: seed with an explicit instruction to ASK for help,
            # not offer it (prevents the persona from playing assistant).
            if script == "banglish":
                contents = (
                    "notun kothopokkhop shuru hocche. apni shahajjo cacchen — "
                    "ekhon apnar lokkho onujayi shohogarirke apnar prosno ba onurodh likhun. "
                    "mone rakbhen: apni shahajjokari nom, apni shahajjo cacchen. "
                    "banglish (romanized bangla) e likhun."
                )
            else:
                contents = (
                    "নতুন কথোপকথন শুরু হচ্ছে। আপনি সাহায্য চাইছেন — "
                    "এখন আপনার লক্ষ্য অনুযায়ী সহকারীকে আপনার প্রশ্ন বা অনুরোধ লিখুন। "
                    "মনে রাখবেন: আপনি সাহায্যকারী নন, আপনি সাহায্য চাইছেন।"
                )
        resp = self._with_retry(lambda: self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.9,
            ),
        ))
        text = (resp.text or "").strip()
        self._track_usage(resp, text)
        return text

    def assistant_turn(
        self,
        system_prompt: str,
        history: list[Message],
        tools: list[ToolDef],
        seed: int,
    ) -> AssistantResponse:
        """Ask Gemini-as-assistant for the next turn. Returns tool calls OR final."""
        self._check_budget(est_tokens=1500)
        declarations = [
            types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters_json_schema=t.parameters_json_schema,
            )
            for t in tools
        ]
        contents = self._history_to_contents(history)

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[types.Tool(function_declarations=declarations)],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    # AUTO lets the model choose between a tool call and a natural-language
                    # answer. ANY forces a tool call every turn, which causes infinite loops
                    # when the model has the result but cannot emit a final answer.
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    ),
                    temperature=0.7,
                ),
            )
        resp = self._with_retry(call)

        # Gemini 3.x models emit an opaque thought_signature on each function-
        # call Part that MUST be replayed when the call is fed back in history.
        # We capture it here by walking the candidate parts (resp.function_calls
        # drops it), then thread it through ToolCall → history reconstruction.
        sig_by_index: dict[int, bytes] = {}
        try:
            parts = resp.candidates[0].content.parts or []
            fc_idx = 0
            for p in parts:
                if getattr(p, "function_call", None):
                    sig = getattr(p, "thought_signature", None)
                    if sig:
                        sig_by_index[fc_idx] = sig
                    fc_idx += 1
        except (IndexError, AttributeError):
            pass

        tool_calls: list[ToolCall] = []
        for i, fc in enumerate(resp.function_calls or []):
            import base64
            sig_b64 = base64.b64encode(sig_by_index[i]).decode() if i in sig_by_index else None
            # Preserve the model-assigned id when present (3.x); fall back to a
            # deterministic id for 2.x models that don't emit one.
            tc_id = getattr(fc, "id", None) or f"call_{seed}_{i}"
            tool_calls.append(ToolCall(
                id=tc_id,
                name=fc.name,
                arguments=dict(fc.args or {}),
                thought_signature_b64=sig_b64,
            ))
        text = (resp.text or "").strip()
        self._track_usage(resp, text)
        return AssistantResponse(text=text, tool_calls=tool_calls)

    def judge(self, trajectory_json: str, rubric_prompt: str) -> Verdict:
        """Score a finished trajectory. Returns a Verdict with 5 JudgeScores."""
        self._check_budget(est_tokens=2000)
        contents = rubric_prompt + "\n\nট্রাজেক্টরি:\n" + trajectory_json

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "dimension": {"type": "string"},
                                        "score": {"type": "integer"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["dimension", "score", "reason"],
                                },
                            }
                        },
                        "required": ["scores"],
                    },
                    temperature=0.2,
                ),
            )
        resp = self._with_retry(call)
        data = json.loads(resp.text or "{}")
        self._track_usage(resp, trajectory_json)
        scores = [JudgeScore(**s) for s in data.get("scores", [])]
        passed = all(s.score >= 4 for s in scores)
        return Verdict(passed=passed, scores=scores)

    def classify_register(self, text: str, script: str = "bengali") -> dict:
        """Classify a user turn's Bangla speech register (tui/tumi/apni).

        Returns ``{"register": str, "confidence": float, "reason": str}``.
        Uses JSON mode + low temperature for deterministic labels.
        """
        self._check_budget(est_tokens=600)
        lang_note = "banglish (romanized bangla)" if script == "banglish" else "বাংলা (bengali script)"
        contents = (
            "Classify the Bangla speech register of this user utterance into exactly "
            "one of three honorific levels:\n"
            "- tui   (intimate: তুই/তোর, tui/tor) — very close friends, siblings. "
            "Verb endings: দিস, করিস, দে, বলিস, রে.\n"
            "- tumi  (familiar: তুমি/তোমাকে, tumi/tomake) — friends, colleagues. "
            "Verb endings: করো, দেখো, দাও, বলো, দেবে.\n"
            "- apni  (formal: আপনি, apni) — elders, strangers, officials. "
            "Verb endings: করবেন, পারবেন, করুন, অনুগ্রহ, দেবেন.\n\n"
            f"Utterance ({lang_note}):\n{text}\n\n"
            "Decide based on pronouns and verb morphology (the primary signal). "
            "If the utterance lacks decisive markers, pick the closest based on "
            "filler words / sentence style and set confidence low."
        )

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "register": {"type": "string", "enum": ["tui", "tumi", "apni"]},
                            "confidence": {"type": "number"},
                            "reason": {"type": "string"},
                        },
                        "required": ["register", "confidence", "reason"],
                    },
                    temperature=0.1,
                ),
            )
        resp = self._with_retry(call)
        data = json.loads(resp.text or "{}")
        data.setdefault("register", "apni")
        data.setdefault("confidence", 0.5)
        data.setdefault("reason", "")
        self._track_usage(resp, text)
        return data

    def classify_register_batch(self, items: list[dict]) -> list[dict]:
        """Classify the speech register of many user turns in ONE API call.

        ``items`` is a list of ``{"id": str, "text": str, "script": str}``.
        Returns a list of ``{"id", "register", "confidence", "reason"}`` in the
        same order. Falls back to per-item classification on any batch failure.
        """
        if not items:
            return []
        self._check_budget(est_tokens=600 * len(items))
        numbered = "\n\n".join(
            f"[{i}] ({it.get('script', 'bengali')}) {it['text']}"
            for i, it in enumerate(items)
        )
        contents = (
            "Classify the Bangla speech register of each numbered user utterance into "
            "exactly one of three honorific levels:\n"
            "- tui   (intimate: তুই/তোর) — very close friends, siblings. "
            "Verb endings: দিস, করিস, দে, বলিস, রে.\n"
            "- tumi  (familiar: তুমি/তোমাকে) — friends, colleagues. "
            "Verb endings: করো, দেখো, দাও, বলো, দেবে.\n"
            "- apni  (formal: আপনি) — elders, strangers, officials. "
            "Verb endings: করবেন, পারবেন, করুন, অনুগ্রহ, দেবেন.\n\n"
            "Decide based on pronouns and verb morphology. Return a JSON object with "
            "a 'results' array, one entry per input, in order. Each entry: "
            "{index, register, confidence (0-1), reason (short)}.\n\n"
            f"Utterances:\n{numbered}"
        )

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema={
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "register": {"type": "string", "enum": ["tui", "tumi", "apni"]},
                                        "confidence": {"type": "number"},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["index", "register", "confidence", "reason"],
                                },
                            }
                        },
                        "required": ["results"],
                    },
                    temperature=0.1,
                ),
            )
        try:
            resp = self._with_retry(call)
            self._track_usage(resp, numbered)
            data = json.loads(resp.text or "{}")
            results = {r["index"]: r for r in data.get("results", [])}
            out = []
            for i, it in enumerate(items):
                r = results.get(i, {})
                out.append({
                    "id": it["id"],
                    "register": r.get("register", "apni"),
                    "confidence": r.get("confidence", 0.5),
                    "reason": r.get("reason", ""),
                })
            return out
        except Exception as e:  # noqa: BLE001
            _log.warning("batch classify failed (%s); falling back to per-item", e)
            return [self._classify_one_fallback(it) for it in items]

    def _classify_one_fallback(self, it: dict) -> dict:
        """Per-item fallback if a batch call fails."""
        try:
            r = self.classify_register(it["text"], script=it.get("script", "bengali"))
            return {"id": it["id"], "register": r["register"],
                    "confidence": r["confidence"], "reason": r.get("reason", "")}
        except Exception:  # noqa: BLE001
            return {"id": it["id"], "register": "apni", "confidence": 0.0, "reason": "fallback"}

    def rewrite_user_turn(self, text: str, target_register: str, script: str = "bengali") -> str:
        """Rewrite a user turn's verb morphology to match ``target_register``.

        Preserves all entities, numbers, IDs, and tool-relevant tokens; changes
        only pronouns, verb endings, fillers, and honorifics.
        """
        self._check_budget(est_tokens=800)
        desc = _REWRITE_TARGETS.get((target_register, script))
        if desc is None:
            # fallback: bengali descriptors
            desc = _REWRITE_TARGETS.get((target_register, "bengali"))
        if desc is None:
            return text  # unknown target — leave unchanged
        lang_note = "banglish (romanized)" if script == "banglish" else "বাংলা লিপিতে (bengali script)"
        contents = (
            f"এই বাক্যটি পুনরায় লিখুন যাতে এটি নিচের register মেনে চলে:\n{desc}\n\n"
            f"নিয়ম (STRICT):\n"
            "- শুধুমাত্র pronoun, verb ending, filler ও honorific পরিবর্তন করুন।\n"
            "- সংখ্যা, নাম, ID, অ্যাকাউন্ট নম্বর, তারিখ, মূল্রমান একদম অপরিবর্তিত রাখুন।\n"
            "- অর্থ ও তথ্য অপরিবর্তিত রাখুন — শুধু speech register পাল্টান।\n"
            f"- উত্তর {lang_note}-এ দিন, শুধু পুনরায় লেখা বাক্যটি (কোনো ব্যাখ্যা নয়)।\n\n"
            f"মূল বাক্য:\n{text}"
        )

        def call() -> types.GenerateContentResponse:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    temperature=0.3,
                ),
            )
        resp = self._with_retry(call)
        out = (resp.text or "").strip()
        self._track_usage(resp, text)
        return out

    @staticmethod
    def _history_to_contents(history: list[Message]) -> list[types.Content]:
        import base64
        import json as _json
        out: list[types.Content] = []
        for m in history:
            if m.role is Role.SYSTEM:
                continue
            # Gemini requires functionResponse parts to carry role="user"; the
            # client supplies the function output. role="model" is rejected with
            # HTTP 400 on the next assistant turn (FIX 1).
            role = "user" if m.role in (Role.TOOL, Role.USER) else "model"
            parts: list[types.Part] = []
            if m.tool_calls:
                for tc in m.tool_calls:
                    fc_part = types.Part(
                        function_call=types.FunctionCall(
                            name=tc.name, args=tc.arguments, id=tc.id,
                        )
                    )
                    # Replay the thought_signature captured from the original
                    # response — required by Gemini 3.x models, ignored by 2.x.
                    if tc.thought_signature_b64:
                        try:
                            fc_part.thought_signature = base64.b64decode(tc.thought_signature_b64)
                        except Exception:  # noqa: BLE001 - signature is best-effort
                            pass
                    parts.append(fc_part)
            if m.role is Role.TOOL:
                # Use the real tool name (not a "tool" placeholder) and pass the
                # result as a parsed object, not a double-encoded JSON string.
                # Find the matching tool call to get the name.
                tc_name = "tool"
                for msg in history:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.id == m.tool_call_id:
                                tc_name = tc.name
                                break
                try:
                    result_obj = _json.loads(m.content or "{}")
                except (ValueError, TypeError):
                    result_obj = {"result": m.content or ""}
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name=tc_name, response=result_obj, id=m.tool_call_id,
                )))
            elif m.content:
                parts.append(types.Part(text=m.content))
            out.append(types.Content(role=role, parts=parts))
        return out
