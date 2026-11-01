from jontro_modal.arms import adapter_dir, label, parse_arm
from jontro_modal.toolparse import parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": "recharge_mobile", "parameters": {"type": "object", "properties": {
    "number": {"type": "string"}, "amount": {"type": "integer"}, "express": {"type": "boolean"}, "tags": {"type": "array"}}}}}]


def test_parse_coerces_by_schema():
    text = ("sure\n<tool_call>\n<function=recharge_mobile>\n<parameter=number>\n01711223344\n</parameter>\n"
            "<parameter=amount>\n50\n</parameter>\n<parameter=express>\nTrue\n</parameter>\n"
            "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>")
    (c,) = parse_tool_calls(text, TOOLS)
    assert c["name"] == "recharge_mobile"
    assert c["arguments"] == {"number": "01711223344", "amount": 50, "express": True, "tags": ["a", "b"]}


def test_parse_multiple_and_none():
    one = "<tool_call>\n<function=a>\n</function>\n</tool_call>"
    assert len(parse_tool_calls(one + "\n" + one, None)) == 2
    assert parse_tool_calls("no call here", TOOLS) == []


def test_bad_number_kept_as_string():
    (c,) = parse_tool_calls("<tool_call>\n<function=recharge_mobile>\n<parameter=amount>\nfifty\n</parameter>\n</function>\n</tool_call>", TOOLS)
    assert c["arguments"]["amount"] == "fifty"


def test_arm_grammar():
    assert parse_arm("run") == ("run", None, 1.0)
    assert parse_arm("run@step_100*0.5") == ("run", "step_100", 0.5)
    assert label("run@step_100*0.5") == "run__step_100_x0.5"
    assert adapter_dir("run@step_100") == "/out/ckpt/run/step_100/adapter"
    assert adapter_dir("run*0.5") == "/out/adapters/run"
