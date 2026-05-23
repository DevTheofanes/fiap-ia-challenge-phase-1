from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.llm.client import LLMResponse
from src.multimodal import audio


def test_transcribe_uses_openai_audio_transcriptions(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    calls = {}

    class _FakeTranscriptions:
        def create(self, **kwargs):
            calls.update(kwargs)
            assert kwargs["file"].read() == b"fake-audio"
            return "transcribed text"

    class _FakeOpenAI:
        def __init__(self, *, api_key):
            calls["api_key"] = api_key
            self.audio = SimpleNamespace(transcriptions=_FakeTranscriptions())

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    transcript = audio.transcribe(audio_path, model="whisper-test")

    assert transcript == "transcribed text"
    assert calls["api_key"] == "openai-key"
    assert calls["model"] == "whisper-test"
    assert calls["response_format"] == "text"


def test_transcribe_rejects_missing_file():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        audio.transcribe("/tmp/does-not-exist.wav")


def test_transcribe_requires_openai_key(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        audio.transcribe(audio_path)


def test_analyze_transcript_builds_required_clinical_prompt():
    calls = {}

    class _FakeClient:
        def generate(self, prompt, *, temperature=0.2, max_tokens=512):
            calls["prompt"] = prompt
            calls["temperature"] = temperature
            calls["max_tokens"] = max_tokens
            return LLMResponse(text="structured report", model="fake", provider="fake")

    result = audio.analyze_transcript("Estou muito cansada e com medo.", client=_FakeClient())

    assert result == "structured report"
    assert calls["temperature"] == 0.1
    assert calls["max_tokens"] == 900
    assert "depressao pos-parto" in calls["prompt"]
    assert "ansiedade" in calls["prompt"]
    assert "violencia" in calls["prompt"]
    assert "fadiga hormonal" in calls["prompt"]
    assert "Estou muito cansada e com medo." in calls["prompt"]


def test_analyze_transcript_uses_configured_client(monkeypatch):
    class _FakeClient:
        def generate(self, prompt, *, temperature=0.2, max_tokens=512):
            return LLMResponse(text="configured client report")

    monkeypatch.setattr(audio, "get_llm_client", lambda: _FakeClient())

    assert audio.analyze_transcript("Relato clinico.") == "configured client report"


def test_analyze_transcript_requires_client(monkeypatch):
    monkeypatch.setattr(audio, "get_llm_client", lambda: None)

    with pytest.raises(RuntimeError, match="No LLM client configured"):
        audio.analyze_transcript("Relato clinico.")


def test_analyze_transcript_rejects_empty_transcript():
    with pytest.raises(ValueError, match="Transcript cannot be empty"):
        audio.analyze_transcript("   ")
