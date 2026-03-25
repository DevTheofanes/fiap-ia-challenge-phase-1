from src.assistant.patient_store import (
    format_patient_context,
    get_patient_source,
    load_patient_record,
)


def test_load_patient_record():
    record = load_patient_record("P-0001")
    assert record is not None
    assert record["patient_id"] == "P-0001"


def test_load_patient_record_missing():
    assert load_patient_record("P-9999") is None


def test_get_patient_source():
    record = load_patient_record("P-0001")
    assert get_patient_source(record) == "patient_record:P-0001"


def test_format_patient_context_order():
    record = load_patient_record("P-0001")
    context = format_patient_context(record)
    assert context.index("Demographics:") < context.index("Chief complaint:")
    assert context.index("Chief complaint:") < context.index("History personal_history:")
    assert context.index("History comorbidities:") < context.index("Medications:")
    assert context.index("ML context:") < context.index("Last updated:")
