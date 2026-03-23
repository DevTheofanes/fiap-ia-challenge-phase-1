import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.assistant.audit_logger import log_interaction


def test_log_interaction_writes_required_fields(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        # Reset the module-level logger so it picks up the patched path
        # (without this, the singleton would still point to the real AUDIT_LOG_PATH)
        import src.assistant.audit_logger as mod
        mod._logger = None

        log_interaction(
            user_query="What is oncology?",
            retrieved_docs=["medquad.txt"],
            model_response="Oncology is the study of cancer.",
            guardrail_triggered=False,
            intent="medical",
            error=None,
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


def test_log_interaction_guardrail_triggered(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        import src.assistant.audit_logger as mod
        mod._logger = None

        log_interaction(
            user_query="What's the weather?",
            retrieved_docs=[],
            model_response="Out of scope.",
            guardrail_triggered=True,
        )

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert records[0]["guardrail_triggered"] is True
    assert records[0]["retrieved_docs"] == []
