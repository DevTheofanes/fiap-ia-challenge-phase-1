import pytest
from src.assistant.guardrails import filter_input, filter_output, _DISCLAIMER

# ── filter_input ──────────────────────────────────────────────

def test_filter_input_empty_query():
    blocked, msg = filter_input("")
    assert blocked is True
    assert msg is not None

def test_filter_input_whitespace_only():
    blocked, msg = filter_input("   ")
    assert blocked is True

def test_filter_input_off_topic_weather():
    blocked, msg = filter_input("What is the weather in São Paulo today?")
    assert blocked is True

def test_filter_input_prescription_request():
    blocked, msg = filter_input("Please prescribe me some antibiotics")
    assert blocked is True

def test_filter_input_medical_passes():
    blocked, msg = filter_input("What are the symptoms of breast cancer?")
    assert blocked is False
    assert msg is None

def test_filter_input_oncology_passes():
    blocked, msg = filter_input("Explain the stages of tumor progression")
    assert blocked is False

# ── filter_output ─────────────────────────────────────────────

def test_filter_output_appends_disclaimer():
    result = filter_output("Some medical answer.")
    assert _DISCLAIMER in result

def test_filter_output_idempotent():
    response_with_disclaimer = "Some medical answer." + _DISCLAIMER
    result = filter_output(response_with_disclaimer)
    assert result.count(_DISCLAIMER) == 1

def test_filter_output_softens_definitive_diagnosis():
    response = "You have cancer and the diagnosis is confirmed."
    result = filter_output(response)
    assert "you have cancer" not in result.lower()
    assert "findings are consistent with possible" in result.lower()
    assert "the preliminary indication is" in result.lower()

def test_filter_output_preserves_normal_response():
    response = "Breast cancer symptoms may include a lump or skin changes."
    result = filter_output(response)
    assert "Breast cancer symptoms may include" in result
