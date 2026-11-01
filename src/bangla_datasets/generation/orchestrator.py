"""Self-play orchestrator: drives the turn-protocol state machine."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from bangla_datasets.gemini.client import AssistantResponse
from bangla_datasets.gemini.prompts import build_assistant_prompt, build_persona_prompt
from bangla_datasets.schema import Message, Persona, Role, ToolDef, Trajectory
from bangla_datasets.tools.registry import ToolRegistry
from bangla_datasets.utils.seeding import derive_seed

_log = logging.getLogger("bangla_datasets.generation")


@dataclass
class SeedPlan:
    """Minimal seed plan consumed by the orchestrator.

    Richer sampling in `generation/seeds.py` constructs these.
    """

    seed: int
    domain: str
    tool_subset: list[ToolDef]
    persona: Persona
    goal: str


class _ClientLike(Protocol):
    """The generation methods the orchestrator drives.

    The real ``GeminiClient`` also exposes ``judge`` (unused here). Only the two
    generation methods are needed, so the Protocol is intentionally narrow.
    """

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


class SelfPlayOrchestrator:
    """Turn-protocol state machine. Intercepts tool calls; simulator injects output.

    The anti-hallucination guarantee: Gemini never produces tool outputs. Every
    function call the assistant emits is intercepted here and executed
    deterministically by the simulator (Task 3), which produces the ``tool`` role
    message. The assistant only ever sees simulator-produced tool results.
    """

    def __init__(
        self,
        client: _ClientLike,
        registry: ToolRegistry | None = None,
        max_turns: int = 12,
    ) -> None:
        self._client = client
        self._registry = registry or ToolRegistry()
        self._max_turns = max_turns

    def generate(self, plan: SeedPlan) -> Trajectory:
        tools = plan.tool_subset
        tool_dicts = [{"name": t.name, "description": t.description} for t in tools]
        # New axes thread through here: system-prompt LANGUAGE drives which
        # assistant prompt we build; conversation SCRIPT flows into the persona
        # prompt + the opening seed. Both default safely for old/golden plans.
        spl = getattr(plan.persona, "system_prompt_language", "bangla") or "bangla"
        script = getattr(plan.persona, "script", "bengali") or "bengali"
        system_prompt = build_assistant_prompt(tool_dicts, language=spl)
        persona_sys = build_persona_prompt(plan.persona, plan.domain, plan.goal)

        messages: list[Message] = []
        sim_seeds: dict[str, int] = {}
        terminated: str | None = None
        sim_counter = 0
        turns_taken = 0

        # Opening user turn.
        opening = self._client.persona_turn(
            persona_sys, history=[], seed=derive_seed(plan.seed, "persona"), script=script,
        )
        messages.append(Message(role=Role.USER, content=opening))

        for turn in range(self._max_turns):
            resp = self._client.assistant_turn(
                system_prompt, history=messages, tools=tools,
                seed=derive_seed(plan.seed, f"asst-{turn}"),
            )
            turns_taken += 1

            if resp.tool_calls:
                # Assistant issued tool call(s). Emit assistant msg + execute each.
                # Gemini never produces the tool output: the simulator does, and it
                # is injected as a `tool` role message in the history the assistant
                # sees on the next turn.
                msg = Message(
                    role=Role.ASSISTANT,
                    content=resp.text or None,
                    tool_calls=resp.tool_calls,
                )
                messages.append(msg)
                for tc in resp.tool_calls:
                    sim_seed = derive_seed(plan.seed, f"sim-{sim_counter}")
                    sim_counter += 1
                    result = self._registry.simulate(tc.name, tc.arguments, sim_seed)
                    sim_seeds[tc.id] = sim_seed
                    messages.append(Message(
                        role=Role.TOOL, content=result, tool_call_id=tc.id,
                    ))
                continue

            # Final answer — terminate.
            messages.append(Message(role=Role.ASSISTANT, content=resp.text))
            terminated = "final"
            break
        else:
            terminated = "max_turns"

        _log.info(
            "trajectory %s: %d turns, terminated=%s", plan.seed, turns_taken, terminated,
        )
        return Trajectory(
            id=f"bd-agentic-{plan.seed:06d}",
            domain=plan.domain,
            persona=plan.persona,
            tools=tools,
            system_prompt=system_prompt,
            messages=messages,
            metadata={
                "turns": turns_taken,
                "tools_used": sorted({
                    tc.name for m in messages if m.tool_calls for tc in m.tool_calls
                }),
                "seed": plan.seed,
                "sim_seeds": sim_seeds,
                "terminated": terminated,
                "persona_goal": plan.goal,
            },
        )
