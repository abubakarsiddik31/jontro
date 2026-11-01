"""OpenAI-compatible client for Gemma 4 / Llama 4 via Together/Groq/etc.

These providers expose the OpenAI Chat Completions API shape, including
function-calling. We convert their output back to our ToolCall schema so
scoring is uniform across models.

Rate limiting: free tiers (Groq especially) enforce requests/min and
tokens/min caps. The client retries with exponential backoff on 429s, and
the runner sleeps between requests (configurable via min_delay).

Note: Gemini (the dataset generator) is deliberately NOT offered here — the
paper keeps a hard generator/evaluator separation. See the design spec.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, RateLimitError

from bangla_datasets.eval.task import build_chat_messages, build_tools_param
from bangla_datasets.schema import ToolCall, Trajectory

_log = logging.getLogger("bangla_datasets.eval")

# Provider base URLs. Override via EVAL_BASE_URL if using another.
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free-tier rate-limit handling. On a 429 we back off exponentially, capped.
MAX_RATE_LIMIT_RETRIES = 6
INITIAL_BACKOFF = 4.0  # seconds
MAX_BACKOFF = 120.0


@dataclass
class Prediction:
    """One model prediction for one eval example.

    ``tool_calls`` holds every call the model emitted — multi-turn Jontro
    trajectories routinely issue several parallel calls per assistant turn, so
    keeping only the first would discard most of the signal. ``tool_call``
    returns the first call for the single-turn path.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_text: str = ""
    error: str | None = None

    @property
    def tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


def parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    """Convert OpenAI-style tool_calls to our ToolCall schema.

    OpenAI returns arguments as a JSON *string*; we parse to dict. On invalid
    JSON we keep the call (so scoring flags it arg-invalid) with empty args.
    """
    out: list[ToolCall] = []
    for item in raw or []:
        fn = item.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        # Some providers (e.g. Llama 3.1) emit arguments=None instead of "{}".
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            try:
                out.append(ToolCall(id=item.get("id", ""), name=name, arguments=arguments))
            except Exception:  # noqa: BLE001 - malformed name (e.g. garbled output) -> skip
                _log.warning("skipping malformed tool call: name=%r", name)
    return out


class EvalClient:
    """Runs a model via an OpenAI-compatible endpoint and returns a Prediction."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        use_tools_param: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.use_tools_param = use_tools_param
        self._client = OpenAI(
            api_key=api_key or os.getenv("EVAL_API_KEY"),
            base_url=base_url or os.getenv("EVAL_BASE_URL"),
            timeout=timeout,
        )

    def _create_with_backoff(self, tag: str, **kwargs: Any):
        """Call the endpoint, retrying 429s with exponential backoff."""
        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                # Honor Retry-After header if present, else exponential backoff.
                wait = getattr(exc, "retry_after", None) or backoff
                wait = min(float(wait), MAX_BACKOFF)
                _log.warning(
                    "rate limited on %s (attempt %d/%d); sleeping %.1fs",
                    tag,
                    attempt + 1,
                    MAX_RATE_LIMIT_RETRIES,
                    wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, MAX_BACKOFF)

    def predict_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tag: str = "",
    ) -> Prediction:
        """Predict the next assistant turn for an arbitrary message prefix.

        This is the primitive both protocols use. ``tools`` may be empty/None
        for a no-tools call. All emitted tool calls are returned.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }
        if self.use_tools_param and tools:
            kwargs["tools"] = tools
        try:
            resp = self._create_with_backoff(tag, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface as Prediction.error
            _log.warning("request failed for %s: %s", tag, exc)
            return Prediction(tool_calls=[], raw_text="", error=str(exc))

        # Some providers (e.g. free-tier hosted models) return responses with no
        # choices at all (empty completion / content filter). Treat as no call.
        if not resp.choices:
            return Prediction(tool_calls=[], raw_text="", error="empty_response")
        choice = resp.choices[0].message
        raw_tcs = (
            [tc.model_dump() for tc in choice.tool_calls] if choice.tool_calls else None
        )
        return Prediction(
            tool_calls=parse_tool_calls(raw_tcs),
            raw_text=choice.content or "",
        )

    def predict(self, traj: Trajectory) -> Prediction:
        """Single-turn prediction: system prompt + first user turn."""
        return self.predict_messages(
            build_chat_messages(traj), build_tools_param(traj), tag=traj.id
        )
