"""Explainer wrapper for model prediction explanations (M4).

Uses Gemini to produce a citation-based 1-2 sentence explanation of why
the assistant gave a particular response.
"""
from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ASSISTANT_LOG_PATH
from src.logging_utils import log_event, setup_json_logger

_PROMPT_TEMPLATE = """\
You are assisting a doctor reviewing a medical assistant's response.
In 1-2 sentences, explain why the assistant gave this answer, \
citing the specific source(s) it used.

Question: {query}
Sources consulted: {sources}
{ml_context_block}
Assistant response (excerpt): {response_excerpt}

Reply in the same language as the question. Be concise."""

_llm: ChatGoogleGenerativeAI | None = None
_logger = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )
    return _llm


def _get_logger():
    global _logger
    if _logger is None:
        _logger = setup_json_logger("explainer", ASSISTANT_LOG_PATH)
    return _logger


def explain_prediction(
    *,
    query: str,
    response: str,
    sources: list[str],
    ml_context: str | None = None,
) -> str:
    """Uses Gemini to produce a 1-2 sentence explanation citing sources.

    Returns empty string if sources list is empty.
    Falls back to template string on API error or when LLM_USE_MOCK=true.
    """
    if not sources:
        return ""

    sources_joined = ", ".join(sources)
    fallback = f"Based on: {sources_joined}."

    if os.getenv("LLM_USE_MOCK", "false").lower() == "true":
        return fallback

    ml_context_block = f"ML Pipeline Context: {ml_context}\n" if ml_context else ""
    response_excerpt = response[:300]

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        sources=sources_joined,
        ml_context_block=ml_context_block,
        response_excerpt=response_excerpt,
    )

    try:
        result = _get_llm().invoke(prompt)
        return result.content.strip()
    except Exception as exc:
        log_event(_get_logger(), "explainer_fallback", error=str(exc), sources=sources)
        return fallback
