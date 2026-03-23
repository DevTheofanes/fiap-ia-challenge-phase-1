import os
import pytest
from unittest.mock import patch, MagicMock

from src.assistant.explainer import explain_prediction


def test_explain_prediction_empty_sources_returns_empty():
    result = explain_prediction(
        query="test",
        response="test response",
        sources=[],
        ml_context=None,
    )
    assert result == ""


def test_explain_prediction_mock_mode_returns_fallback(monkeypatch):
    monkeypatch.setenv("LLM_USE_MOCK", "true")
    result = explain_prediction(
        query="What is cancer?",
        response="Cancer is a disease.",
        sources=["medquad.txt", "pubmed.txt"],
        ml_context=None,
    )
    assert result == "Based on: medquad.txt, pubmed.txt."


def test_explain_prediction_exception_returns_fallback(monkeypatch):
    monkeypatch.setenv("LLM_USE_MOCK", "false")
    # Reset the cached LLM so it tries to call and fails
    import src.assistant.explainer as mod
    original_llm = mod._llm
    mod._llm = None

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API error")

    with patch("src.assistant.explainer._get_llm", return_value=mock_llm):
        result = explain_prediction(
            query="What is oncology?",
            response="Oncology is the study of cancer.",
            sources=["source.txt"],
            ml_context=None,
        )

    assert result == "Based on: source.txt."
    mod._llm = original_llm


def test_explain_prediction_mock_with_ml_context(monkeypatch):
    monkeypatch.setenv("LLM_USE_MOCK", "true")
    result = explain_prediction(
        query="test",
        response="test",
        sources=["file.txt"],
        ml_context="75% probability of malignancy",
    )
    assert result == "Based on: file.txt."
