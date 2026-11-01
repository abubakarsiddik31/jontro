"""OpenAI Chat Completion Tool-Calling writer.

Standard format consumed by Hugging Face TRL (SFTTrainer), Unsloth,
vLLM, SGLang, and apply_chat_template().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bangla_datasets.schema import Role, Trajectory


def to_openai_format(trajectory: Trajectory) -> dict[str, Any]:
    """Convert a Trajectory into standard OpenAI Chat Completion / TRL tool format."""
    messages = []

    # 1. System prompt message
    if trajectory.system_prompt:
        messages.append({
            "role": "system",
            "content": trajectory.system_prompt,
        })

    # 2. Conversation turns
    for m in trajectory.messages:
        if m.role is Role.SYSTEM:
            continue
        elif m.role is Role.USER:
            messages.append({
                "role": "user",
                "content": m.content or "",
            })
        elif m.role is Role.TOOL:
            messages.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id or "call_unknown",
                "content": m.content or "",
            })
        elif m.role is Role.ASSISTANT:
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": m.content if m.content else None,
            }
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            messages.append(entry)

    # 3. Available tools schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_json_schema,
            },
        }
        for t in trajectory.tools
    ]

    return {
        "id": trajectory.id,
        "domain": trajectory.domain,
        "persona": trajectory.persona.model_dump(),
        "messages": messages,
        "tools": tools,
        "metadata": trajectory.metadata,
    }


def write_openai_jsonl(trajectories: list[Trajectory], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in trajectories:
            formatted = to_openai_format(t)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
