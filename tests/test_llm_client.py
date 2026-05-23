from __future__ import annotations

import sys
from types import SimpleNamespace

from src.llm import client as client_mod
from src.llm.client import GeminiClient, MockClient, OpenAIClient


def _clear_llm_env(monkeypatch) -> None:
    monkeypatch.setattr(client_mod, "_load_env_file", lambda env_path=None: None)
    for key in (
        "LLM_USE_MOCK",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def test_get_llm_client_mock_takes_precedence(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_USE_MOCK", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    client = client_mod.get_llm_client()

    assert isinstance(client, MockClient)


def test_get_llm_client_returns_openai_when_key_is_defined(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    client = client_mod.get_llm_client()

    assert isinstance(client, OpenAIClient)
    assert client.api_key == "openai-key"
    assert client.model == "gpt-test"


def test_get_llm_client_returns_gemini_when_openai_key_is_missing(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")

    client = client_mod.get_llm_client()

    assert isinstance(client, GeminiClient)
    assert client.api_key == "gemini-key"
    assert client.model == "gemini-test"


def test_get_llm_client_returns_none_without_provider_keys(monkeypatch):
    _clear_llm_env(monkeypatch)

    assert client_mod.get_llm_client() is None


def test_openai_client_generate_uses_responses_api(monkeypatch):
    calls = {}

    class _FakeResponses:
        def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(output_text="clinical response")

    class _FakeOpenAI:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.responses = _FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    client = OpenAIClient(api_key="openai-key", model="gpt-test")
    response = client.generate("Summarize case", temperature=0.1, max_tokens=123)

    assert response.text == "clinical response"
    assert response.model == "gpt-test"
    assert response.provider == "openai"
    assert calls == {
        "api_key": "openai-key",
        "model": "gpt-test",
        "input": "Summarize case",
        "temperature": 0.1,
        "max_output_tokens": 123,
    }
