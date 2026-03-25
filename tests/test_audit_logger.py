import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

import src.assistant.audit_logger as audit_logger_mod
from src.assistant.audit_logger import log_interaction


@pytest.fixture(autouse=True)
def reset_audit_logger():
    """Reset the audit logger singleton and clear logging handlers between tests."""
    yield
    audit_logger = logging.getLogger("audit")
    for handler in list(audit_logger.handlers):
        handler.close()
        audit_logger.removeHandler(handler)
    audit_logger_mod._logger = None


def test_log_interaction_writes_required_fields(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        log_interaction(
            user_query="What is oncology?",
            retrieved_docs=["medquad.txt"],
            model_response="Oncology is the study of cancer.",
            guardrail_triggered=False,
            intent="medical",
            error=None,
            patient_id="P-0001",
            patient_context_used=True,
            kb_sources=["medquad.txt"],
            patient_source="patient_record:P-0001",
        )

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert len(records) == 1
    rec = records[0]
    assert "timestamp" in rec
    assert rec["user_query"] == "What is oncology?"
    assert rec["retrieved_docs"] == ["medquad.txt"]
    assert rec["model_response"] == "Oncology is the study of cancer."
    assert rec["guardrail_triggered"] is False
    assert rec["intent"] == "medical"
    assert rec["patient_id"] == "P-0001"
    assert rec["patient_context_used"] is True
    assert rec["kb_sources"] == ["medquad.txt"]
    assert rec["patient_source"] == "patient_record:P-0001"


def test_log_interaction_guardrail_triggered(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        log_interaction(
            user_query="What's the weather?",
            retrieved_docs=[],
            model_response="Out of scope.",
            guardrail_triggered=True,
        )

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert records[0]["guardrail_triggered"] is True
    assert records[0]["retrieved_docs"] == []
