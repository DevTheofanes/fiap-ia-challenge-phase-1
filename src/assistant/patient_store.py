"""Structured synthetic patient-record store for the medical assistant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PATIENTS_DIR


def load_patient_record(patient_id: str) -> dict[str, Any] | None:
    """Load one synthetic patient record by patient_id."""
    if not patient_id:
        return None
    path = PATIENTS_DIR / f"{patient_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_patient_source(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    patient_id = record.get("patient_id")
    if not patient_id:
        return None
    return f"patient_record:{patient_id}"


def format_patient_context(record: dict[str, Any]) -> str:
    """Render patient context in a deterministic field order for prompting."""
    demographics = record.get("demographics", {})
    history = record.get("history", {})
    recent_exams = record.get("recent_exams", {})
    recent_labs = record.get("recent_labs", {})
    ml_context = record.get("ml_context", {})

    def _fmt_mapping(title: str, mapping: dict[str, Any]) -> str:
        if not mapping:
            return f"{title}: none"
        parts = [f"{key}={value}" for key, value in mapping.items()]
        return f"{title}: " + "; ".join(parts)

    def _fmt_list(title: str, items: list[Any]) -> str:
        if not items:
            return f"{title}: none"
        return f"{title}: " + ", ".join(str(item) for item in items)

    sections = [
        _fmt_mapping("Demographics", demographics),
        f"Chief complaint: {record.get('chief_complaint', 'none')}",
        _fmt_list("History personal_history", history.get("personal_history", [])),
        _fmt_list("History family_history", history.get("family_history", [])),
        _fmt_list("History comorbidities", history.get("comorbidities", [])),
        _fmt_list("Medications", record.get("current_medications", [])),
        _fmt_mapping("Recent exams", recent_exams),
        _fmt_mapping("Recent labs", recent_labs),
        f"Imaging summary: {record.get('imaging_summary', 'none')}",
        f"Clinician notes: {record.get('clinician_notes', 'none')}",
        _fmt_mapping("ML context", ml_context),
        f"Last updated: {record.get('last_updated', 'unknown')}",
    ]
    return "\n".join(sections)
