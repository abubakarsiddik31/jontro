"""Unit tests for InteractiveOrchestrator and Anti-Tui gate."""
from __future__ import annotations

from dataclasses import dataclass

from bangla_datasets.gemini.client import AssistantResponse
from bangla_datasets.generation.interactive_orchestrator import (
    InteractiveOrchestrator,
    MultiTurnSeedPlan,
    contains_assistant_tui,
    is_user_signoff,
)
from bangla_datasets.schema import Message, Persona, Role, ToolCall
from bangla_datasets.tools.catalog import BOOK_TICKET, GET_TRAIN_FARE, SEARCH_TRAINS
from bangla_datasets.validation.heuristics import validate_multi_turn_trajectory


def test_contains_assistant_tui() -> None:
    assert contains_assistant_tui("তুই কেমন আছিস?") is True
    assert contains_assistant_tui("তোর টিকিট কাটা হয়েছে।") is True
    assert contains_assistant_tui("টাকাটা বিকাশ থেকে পাঠিয়ে দিস।") is True
    assert contains_assistant_tui("tui kothay jabi?") is True
    assert contains_assistant_tui("tor ticket done") is True

    # Polite words MUST pass
    assert contains_assistant_tui("আপনার টিকিট নিশ্চিত করা হয়েছে।") is False
    assert contains_assistant_tui("আপনি কোন ট্রেনে যেতে চাচ্ছেন?") is False
    assert contains_assistant_tui("Apnar ticket confirm kora hoyeche.") is False
    assert contains_assistant_tui("Dhonnobad! Bhalo thakben.") is False


def test_is_user_signoff() -> None:
    # Explicit farewells
    assert is_user_signoff("অনেক ধন্যবাদ ভাই, সব কাজ হয়ে গেছে!") is True
    assert is_user_signoff("থ্যাংকস রে, আর কিছু লাগবে না।") is True
    assert is_user_signoff("Thanks a lot! Shob thik ache.") is True

    # Terse natural acks
    assert is_user_signoff("ঠিক আছে") is True
    assert is_user_signoff("ওকে, পেয়েছি") is True
    assert is_user_signoff("thik ache") is True
    assert is_user_signoff("ok") is True

    # Cancellations / Decisions not to proceed
    assert is_user_signoff("তাহলে থাক, লাগবে না আর") is True
    assert is_user_signoff("আজকে বাদ দাও, অন্য সময় দেখব") is True
    assert is_user_signoff("dam beshi, thak lagbe na") is True

    # User asking a question is NOT a signoff
    assert is_user_signoff("ধন্যবাদ, কিন্তু ট্রেনের ভাড়া কত পড়বে?") is False
    assert is_user_signoff("কালকের ট্রেনের টিকিট আছে?") is False


@dataclass
class FakeMultiTurnClient:
    """Simulates a multi-turn conversation:
    Turn 1 (User): asks for trains
    Turn 2 (Assistant): calls search_trains, then presents options
    Turn 3 (User): selects train and asks to book
    Turn 4 (Assistant): calls book_ticket, then gives confirmation
    Turn 5 (User): thanks assistant
    Turn 6 (Assistant): polite closing
    """

    def __init__(self) -> None:
        self.asst_call_count = 0
        self.user_turn_count = 0

    def persona_turn(self, system: str, history: list[Message], seed: int, **kw: object) -> str:
        self.user_turn_count += 1
        if not history:
            return "ঢাকা থেকে চট্টগ্রাম যাওয়ার ট্রেনের টিকিট লাগবে।"
        if self.user_turn_count == 2:
            return "সুবর্ণা এক্সপ্রেসে স্নিগ্ধা একটা টিকিট কেটে দেন, নাম রহিম।"
        return "অনেক ধন্যবাদ, আমার কাজ হয়ে গেছে!"

    def assistant_turn(
        self, system: str, history: list[Message], tools: list[object], seed: int,
    ) -> AssistantResponse:
        self.asst_call_count += 1
        # Step 1: user asked for trains -> assistant calls search_trains
        if self.asst_call_count == 1:
            return AssistantResponse(
                text="আমি ট্রেন দেখছি",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_trains",
                        arguments={"from": "ঢাকা", "to": "চট্টগ্রাম", "date": "2026-07-20"},
                    )
                ],
            )
        # Step 2: after search_trains -> assistant presents options to user
        if self.asst_call_count == 2:
            return AssistantResponse(
                text="সুবর্ণা ও মহানগর এক্সপ্রেসে আসন খালি আছে। আপনি কোনটি বুক করতে চান?",
                tool_calls=[],
            )
        # Step 3: user chose Subarna -> assistant calls book_ticket
        if self.asst_call_count == 3:
            return AssistantResponse(
                text="বুকিং করছি",
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="book_ticket",
                        arguments={
                            "train_id": "SUBARNA-701",
                            "seat_class": "snigdha",
                            "passenger_name": "রহিম",
                            "seat_count": 1,
                        },
                    )
                ],
            )
        # Step 4: after booking -> confirms PNR to user
        if self.asst_call_count == 4:
            return AssistantResponse(
                text="আপনার টিকিট বুকিং সম্পন্ন হয়েছে! আর কোনো সাহায্য লাগবে?",
                tool_calls=[],
            )
        # Step 5: user said thanks -> polite sign-off
        return AssistantResponse(
            text="আপনাকে অনেক ধন্যবাদ! ভালো থাকবেন ও নিরাপদ ভ্রমণ কামনা করছি।",
            tool_calls=[],
        )


def test_interactive_orchestrator_multi_turn_flow() -> None:
    plan = MultiTurnSeedPlan(
        seed=100,
        domain="railway_booking",
        tool_subset=[SEARCH_TRAINS, GET_TRAIN_FARE, BOOK_TICKET],
        persona=Persona(register="apni", script="bengali"),
        goal="ঢাকা থেকে চট্টগ্রাম ট্রেনের টিকিট কাটা",
        min_user_turns=2,
        max_user_turns=4,
    )
    client = FakeMultiTurnClient()
    orch = InteractiveOrchestrator(client=client)
    traj = orch.generate(plan)

    user_msgs = [m for m in traj.messages if m.role == Role.USER]
    asst_msgs = [m for m in traj.messages if m.role == Role.ASSISTANT]
    tool_msgs = [m for m in traj.messages if m.role == Role.TOOL]

    # Verify multi-turn depth
    assert len(user_msgs) >= 2
    assert len(asst_msgs) >= 2

    # Verify tool execution
    assert len(tool_msgs) >= 2
    tool_names = traj.metadata["tools_used"]
    assert "search_trains" in tool_names
    assert "book_ticket" in tool_names

    # Verify natural termination
    assert traj.metadata["terminated"] == "natural_completion"

    # Verify quality gate
    verdict = validate_multi_turn_trajectory(traj, min_user_turns=2, min_tools=2)
    assert verdict.passed is True
