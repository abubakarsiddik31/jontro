"""Pydantic schemas for trajectories, tools, and validation verdicts."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bangla_datasets.utils.script import has_bangla_key, is_ascii_identifier


class LanguageBoundaryError(ValueError):
    """Raised when tool/function names or keys violate the English-only rule."""


class Role(str, Enum):  # noqa: UP042 - brief mandates str,Enum; StrEnum changes str() repr
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Persona(BaseModel):
    """A simulated user profile driving the conversation."""

    model_config = ConfigDict(extra="allow")

    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = None  # "urban" | "rural"
    region: str | None = None  # "bangladesh"
    profession: str | None = None
    register: str = Field(default="apni")  # intimate(tui) | familiar(tumi) | formal(apni)
    tech_literacy: str = Field(default="medium")  # "low" | "medium" | "high"
    # Script of the *conversation* layer (user + assistant turns). The tool/code
    # layer is always ASCII regardless of this setting. Old/golden data without
    # these fields round-trips via extra="allow" + these defaults.
    script: str = Field(default="bengali")  # "bengali" | "banglish"
    # Language the *system prompt* is written in (production LLMs usually ship
    # English system prompts; Bangla here is for diversity).
    system_prompt_language: str = Field(default="bangla")  # "bangla" | "english"

    @field_validator("register")
    @classmethod
    def _valid_register(cls, v: str) -> str:
        if v not in {"tui", "tumi", "apni"}:
            raise ValueError("register must be 'tui', 'tumi', or 'apni'")
        return v

    @field_validator("script")
    @classmethod
    def _valid_script(cls, v: str) -> str:
        if v not in {"bengali", "banglish"}:
            raise ValueError("script must be 'bengali' or 'banglish'")
        return v

    @field_validator("system_prompt_language")
    @classmethod
    def _valid_spl(cls, v: str) -> str:
        if v not in {"bangla", "english"}:
            raise ValueError("system_prompt_language must be 'bangla' or 'english'")
        return v


class ToolDef(BaseModel):
    """A tool/function schema exposed to the assistant. Names are English."""

    name: str
    description: str
    parameters_json_schema: dict[str, Any]

    @field_validator("name")
    @classmethod
    def _ascii_name(cls, v: str) -> str:
        if not is_ascii_identifier(v):
            raise LanguageBoundaryError(
                f"Tool name must be ASCII identifier (English/code layer): {v!r}"
            )
        return v

    @field_validator("parameters_json_schema")
    @classmethod
    def _ascii_keys(cls, schema: dict[str, Any]) -> dict[str, Any]:
        def check(node: Any) -> None:  # noqa: ANN401 - genuinely dynamic JSON
            if isinstance(node, dict):
                for k, val in node.items():
                    if isinstance(k, str) and has_bangla_key(k):
                        raise LanguageBoundaryError(
                            f"Parameter key must be ASCII (English/code layer): {k!r}"
                        )
                    check(val)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(schema)
        return schema


class ToolCall(BaseModel):
    """A single tool call issued by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]
    # Gemini 3.x models emit an opaque thought_signature on function-call parts
    # that MUST be replayed when the call is fed back in history. Missing it
    # triggers HTTP 400 INVALID_ARGUMENT on the next assistant turn. Stored as
    # base64 so the JSONL trajectory stays text-only and round-trips cleanly.
    # None on older models (2.x) and on assistant turns that emit text only.
    thought_signature_b64: str | None = None

    @field_validator("name")
    @classmethod
    def _ascii_name(cls, v: str) -> str:
        if not is_ascii_identifier(v):
            raise LanguageBoundaryError(f"Tool call name must be ASCII: {v!r}")
        return v


class Message(BaseModel):
    """One turn in a conversation. Tool messages carry tool_call_id."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> Message:
        if self.role is Role.TOOL:  # noqa: SIM102 - brief nests the two checks deliberately
            if self.tool_call_id is None:
                raise ValueError("tool role requires tool_call_id")
        if self.tool_calls is not None and self.role is not Role.ASSISTANT:
            raise ValueError("only assistant messages may carry tool_calls")
        return self


class Trajectory(BaseModel):
    """A complete multi-turn agentic conversation."""

    id: str
    domain: str
    persona: Persona
    tools: list[ToolDef]
    system_prompt: str
    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _first_turn_is_user(self) -> Trajectory:
        non_system = [m for m in self.messages if m.role is not Role.SYSTEM]
        if non_system and non_system[0].role is not Role.USER:
            raise ValueError("first non-system message must be a user turn")
        return self


class JudgeScore(BaseModel):
    """One dimension of the judge's rubric, scored 1-5."""

    dimension: str
    score: int = Field(ge=1, le=5)
    reason: str


class Verdict(BaseModel):
    """Validation result for one trajectory."""

    trajectory_id: str = ""
    passed: bool
    stage_failed: str | None = None
    scores: list[JudgeScore] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    hash: str = ""
