import pytest

from bangla_datasets.gemini.client import AssistantResponse, GeminiClient


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiClient()


def test_client_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    c = GeminiClient()
    assert c._api_key == "test-key"


def test_assistant_response_parse_final() -> None:
    r = AssistantResponse(text="final answer", tool_calls=[])
    assert r.is_final


def test_assistant_response_parse_tool_call() -> None:
    from bangla_datasets.schema import ToolCall

    r = AssistantResponse(text="", tool_calls=[ToolCall(id="c1", name="f", arguments={})])
    assert not r.is_final
