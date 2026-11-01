"""Multi-turn tool-call eval task construction.

A Jontro trajectory is a chained conversation::

    user -> assistant(calls) -> tool -> assistant(calls) -> tool -> ... -> user

We score it at each **decision point** — an assistant message carrying one or
more tool calls. The single-turn harness (``eval/task.py``) truncates a
trajectory to its first call; this module is the multi-turn replacement and
keeps every step.

Two protocols are built here and driven by ``eval/multiturn_runner.py``:

- ``teacher_forced`` — at decision point *i* the model sees the **gold** prefix
  and must predict that step's call set. Errors do not compound, which isolates
  per-step decision quality and yields the per-turn-index curve.
- ``interactive`` — the model drives every step; after each predicted call the
  deterministic simulator produces the result, which is fed back. Errors
  compound, so this measures end-to-end task completion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from bangla_datasets.eval.task import build_tools_param
from bangla_datasets.schema import Message, Role, ToolCall, Trajectory


@dataclass
class DecisionPoint:
    """One assistant tool-call step in a trajectory."""

    index: int  # 0-based position among decision points
    message_index: int  # index into ``traj.messages``
    gold_calls: list[ToolCall]
    # Gold user turns since the previous decision point, in order. Only the
    # interactive protocol uses these (teacher forcing reads the message slice).
    pending_user: list[Message] = field(default_factory=list)


def iter_decision_points(traj: Trajectory) -> list[DecisionPoint]:
    """Every assistant message that carries >= 1 tool call, in order.

    Parallel calls within one assistant message are a single decision point
    (the model emits them together); consecutive assistant messages that each
    call tools are separate points.
    """
    out: list[DecisionPoint] = []
    pending: list[Message] = []
    for i, m in enumerate(traj.messages):
        if m.role is Role.USER:
            pending.append(m)
        elif m.role is Role.ASSISTANT and m.tool_calls:
            out.append(DecisionPoint(len(out), i, list(m.tool_calls), pending))
            pending = []
    return out


def message_to_openai(m: Message) -> dict:
    """Convert one stored message to the OpenAI chat-completions shape.

    ``thought_signature_b64`` is dropped: it is a Gemini replay token, and the
    eval providers are OpenAI-compatible (Gemini is deliberately excluded to
    keep the generator/evaluator separation).
    """
    if m.role is Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id or "",
            "content": m.content or "",
        }
    if m.role is Role.ASSISTANT:
        out: dict = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            out["tool_calls"] = [openai_call(tc, i) for i, tc in enumerate(m.tool_calls)]
        return out
    if m.role is Role.SYSTEM:
        return {"role": "system", "content": m.content or ""}
    return {"role": "user", "content": m.content or ""}


def openai_call(tc: ToolCall, index: int) -> dict:
    """One tool call in the OpenAI function-calling shape (stable fallback id)."""
    return {
        "id": tc.id or f"call_{index}",
        "type": "function",
        "function": {
            "name": tc.name,
            "arguments": json.dumps(tc.arguments or {}, ensure_ascii=False),
        },
    }


def build_step_messages(traj: Trajectory, dp: DecisionPoint) -> list[dict]:
    """System prompt + the stored prefix strictly before *dp*.

    The prefix ends at the user turn(s) and tool results that precede the step,
    so the model must produce the *next* assistant turn.
    """
    msgs: list[dict] = [{"role": "system", "content": traj.system_prompt}]
    msgs += [message_to_openai(m) for m in traj.messages[: dp.message_index]]
    return msgs


def build_step_tools(traj: Trajectory) -> list[dict]:
    """OpenAI-style tools list offered at every step of *traj*."""
    return build_tools_param(traj)


def gold_step_names(traj: Trajectory) -> list[list[str]]:
    """The gold tool-name set at each decision point (for inspection/tests)."""
    return [[c.name for c in dp.gold_calls] for dp in iter_decision_points(traj)]


def simulate_seed(traj_id: str, step: int, call_index: int) -> int:
    """Deterministic simulator seed for a *predicted* call.

    The gold corpus stores a per-call ``sim_seeds`` map; predicted calls have no
    such entry, so the interactive protocol derives a stable seed from the
    trajectory id and call position. This keeps replay byte-reproducible.
    """
    key = f"{traj_id}:{step}:{call_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
