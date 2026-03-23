"""Audit logger for medical assistant interactions (M4).

Logs all interactions to artifacts/logs/audit.jsonl with:
timestamp, user_query, retrieved_docs, model_response, guardrail_triggered.
"""
from __future__ import annotations

import logging

from src.config import AUDIT_LOG_PATH
from src.logging_utils import log_event, setup_json_logger

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_json_logger("audit", AUDIT_LOG_PATH)
    return _logger


def log_interaction(
    *,
    user_query: str,
    retrieved_docs: list[str],
    model_response: str,
    guardrail_triggered: bool,
    intent: str | None = None,
    error: str | None = None,
) -> None:
    """Appends one structured record to artifacts/logs/audit.jsonl.

    Required fields: timestamp, user_query, retrieved_docs,
                     model_response, guardrail_triggered.
    Optional fields: intent, error.
    """
    log_event(
        _get_logger(),
        "audit",
        user_query=user_query,
        retrieved_docs=retrieved_docs,
        model_response=model_response,
        guardrail_triggered=guardrail_triggered,
        intent=intent,
        error=error,
    )
