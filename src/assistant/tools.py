"""Helper tools for the medical assistant."""
from __future__ import annotations

from src.assistant.patient_store import load_patient_record


def lookup_patient_record(patient_id: str):
    """Return one structured patient record by id."""
    return load_patient_record(patient_id)
