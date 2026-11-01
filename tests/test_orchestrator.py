"""Tests for the self-play orchestrator (Task 5).

These use FAKE clients (no network) to exercise the turn-protocol state machine
directly: final-answer termination, tool-then-final, and max-turns cap.
"""
from __future__ import annotations

from dataclasses import dataclass

from bangla_datasets.gemini.client import AssistantResponse
from bangla_datasets.generation.orchestrator import SeedPlan, SelfPlayOrchestrator
from bangla_datasets.schema import Message, Persona, Role, ToolCall
from bangla_datasets.tools.catalog import SEARCH_TRAINS


@dataclass
class FakeAssistantFinal:
    """Fake client whose assistant immediately returns a final answer."""

    def persona_turn(self, system: str, history: list[Message], seed: int, **kw: object) -> str:
        return "আমি ঢাকা থেকে চট্টগ্রাম ট্রেন চাই।"

    def assistant_turn(
        self, system: str, history: list[Message], tools: list[object], seed: int,
    ) -> AssistantResponse:
        return AssistantResponse(text="ঠিক আছে, ট্রেন খোঁজা হলো।", tool_calls=[])

    def judge(self, *a: object, **k: object) -> None:
        pass


def test_orchestrator_final_answer_terminates() -> None:
    plan = SeedPlan(
        seed=1, domain="railway_booking",
        tool_subset=[SEARCH_TRAINS], persona=Persona(), goal="book train",
    )
    orch = SelfPlayOrchestrator(client=FakeAssistantFinal())  # type: ignore[arg-type]
    traj = orch.generate(plan)
    assert traj.id
    assert traj.messages[0].role in (Role.SYSTEM, Role.USER)
    assert any(m.role is Role.USER for m in traj.messages)
    assert traj.metadata["turns"] >= 1


class FakeAssistantToolThenFinal:
    """Fake client: first call -> tool_call, second call -> final."""

    def __init__(self) -> None:
        self._n = 0

    def persona_turn(self, system: str, history: list[Message], seed: int, **kw: object) -> str:
        return "ঢাকা থেকে চট্টগ্রাম ট্রেন দেখান।"

    def assistant_turn(
        self, system: str, history: list[Message], tools: list[object], seed: int,
    ) -> AssistantResponse:
        self._n += 1
        if self._n == 1:
            return AssistantResponse(
                text="খুঁজছি",
                tool_calls=[
                    ToolCall(
                        id="c1", name="search_trains",
                        arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"},
                    )
                ],
            )
        return AssistantResponse(text="৩টি ট্রেন পাওয়া গেছে।", tool_calls=[])

    def judge(self, *a: object, **k: object) -> None:
        pass


def test_orchestrator_executes_tool_then_final() -> None:
    plan = SeedPlan(
        seed=2, domain="railway_booking",
        tool_subset=[SEARCH_TRAINS], persona=Persona(), goal="find trains",
    )
    orch = SelfPlayOrchestrator(client=FakeAssistantToolThenFinal())  # type: ignore[arg-type]
    traj = orch.generate(plan)
    # Should have: user, assistant(tool_call), tool(result), assistant(final)
    roles = [m.role for m in traj.messages]
    assert Role.TOOL in roles
    tool_msg = next(m for m in traj.messages if m.role is Role.TOOL)
    assert tool_msg.tool_call_id == "c1"


class ForeverTool:
    """Fake client that always returns a tool call (never a final answer)."""

    def persona_turn(self, system: str, history: list[Message], seed: int, **kw: object) -> str:
        return "আরও"

    def assistant_turn(
        self, system: str, history: list[Message], tools: list[object], seed: int,
    ) -> AssistantResponse:
        return AssistantResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="cx", name="search_trains",
                    arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"},
                )
            ],
        )

    def judge(self, *a: object, **k: object) -> None:
        pass


def test_orchestrator_max_turns_cap() -> None:
    plan = SeedPlan(
        seed=3, domain="railway_booking",
        tool_subset=[SEARCH_TRAINS], persona=Persona(), goal="x",
    )
    orch = SelfPlayOrchestrator(client=ForeverTool(), max_turns=4)  # type: ignore[arg-type]
    traj = orch.generate(plan)
    assert traj.metadata["turns"] <= 4
    assert traj.metadata.get("terminated") == "max_turns"
