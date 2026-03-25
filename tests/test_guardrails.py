from src.assistant.guardrails import (
    _DISCLAIMER,
    _INSUFFICIENT_CONTEXT,
    _PRESCRIPTION_REFUSAL,
    filter_input,
    filter_output,
    has_guardrail_signal,
)


def test_filter_input_empty_query():
    blocked, msg = filter_input("")
    assert blocked is True
    assert msg is not None


def test_filter_input_off_topic_weather():
    blocked, msg = filter_input("What is the weather in São Paulo today?")
    assert blocked is True
    assert msg is not None


def test_filter_input_prescription_request():
    blocked, msg = filter_input("Please prescribe me some antibiotics")
    assert blocked is True
    assert _PRESCRIPTION_REFUSAL in msg


def test_filter_input_medical_passes():
    blocked, msg = filter_input("What are the symptoms of breast cancer?")
    assert blocked is False
    assert msg is None


def test_filter_output_appends_disclaimer():
    result = filter_output("Some medical answer.", has_context=True)
    assert _DISCLAIMER in result


def test_filter_output_idempotent():
    response_with_disclaimer = "Some medical answer." + _DISCLAIMER
    result = filter_output(response_with_disclaimer, has_context=True)
    assert result.count(_DISCLAIMER) == 1


def test_filter_output_softens_definitive_diagnosis():
    response = "You have cancer and the diagnosis is confirmed."
    result = filter_output(response, has_context=True)
    assert "you have cancer" not in result.lower()
    assert "findings are consistent with possible" in result.lower()
    assert "the preliminary indication is" in result.lower()


def test_filter_output_replaces_direct_treatment_language():
    response = "Start tamoxifen 20 mg daily immediately."
    result = filter_output(response, has_context=True)
    assert _PRESCRIPTION_REFUSAL in result


def test_filter_output_insufficient_context():
    result = filter_output("Any answer.", has_context=False)
    assert _INSUFFICIENT_CONTEXT in result


def test_has_guardrail_signal_detects_change():
    assert has_guardrail_signal("Start drug 10 mg daily", filter_output("Start drug 10 mg daily", has_context=True)) is True
