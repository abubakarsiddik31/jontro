"""Parse Qwen3.5 / qwen3_coder tool calls from generated text, coercing values by the tool schema.

    <tool_call>
    <function=search_doctors>
    <parameter=specialty>
    gastroenterologist
    </parameter>
    </function>
    </tool_call>
"""
from __future__ import annotations

import json
import re

CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict | None):
    t = (schema or {}).get("type")
    try:
        if t == "integer":
            return int(value.strip())
        if t == "number":
            return float(value.strip())
        if t == "boolean":
            return value.strip().lower() == "true"
        if t in ("array", "object"):
            return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> list[dict]:
    props: dict[str, dict] = {}
    for t in tools or []:
        fn = t.get("function", t)
        props[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for i, m in enumerate(CALL.finditer(text)):
        name, body = m.group(1), m.group(2)
        schema = props.get(name, {})
        args = {k: _coerce(v, schema.get(k)) for k, v in PARAM.findall(body)}
        calls.append({"id": f"call_{i}", "name": name, "arguments": args})
    return calls
