"""Input/output guardrails for the medical assistant (M4).

- filter_input: cheap regex pre-filter before LLM classification
- filter_output: disclaimer injection + definitive diagnosis softening
"""
from __future__ import annotations

import re

_DISCLAIMER = (
    "\n\n⚠ Esta informação é educacional e não substitui consulta médica profissional."
)

_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(weather|clima|temperatura|sports|futebol|soccer|football|stock|bolsa|bitcoin|"
    r"crypto|recipe|receita culin|movie|filme|music|m[uú]sica|política|politics|"
    r"lottery|loteria|horoscope|hor[oó]scopo)\b",
    re.IGNORECASE,
)

_PRESCRIPTION_PATTERNS = re.compile(
    r"\b(prescri(be|va|va-me)|give me (medication|medicine|drug|pills?)|"
    r"me d[êe] (rem[eé]dio|medica[çc][aã]o)|quero (comprar|tomar) (rem[eé]dio|medica[çc][aã]o))\b",
    re.IGNORECASE,
)

_TREATMENT_PATTERNS = re.compile(
    r"\b(start|take|use|prescribe|administer)\b.{0,40}\b(mg|tablet|capsule|dose|daily|twice daily)\b",
    re.IGNORECASE,
)

_DEFINITIVE_DIAGNOSIS_PATTERNS = [
    (
        re.compile(r"\byou have (cancer|tumor|malignancy|carcinoma)\b", re.IGNORECASE),
        "findings are consistent with possible",
    ),
    (
        re.compile(r"\bthe diagnosis is\b", re.IGNORECASE),
        "the preliminary indication is",
    ),
    (
        re.compile(r"\byou are diagnosed with\b", re.IGNORECASE),
        "findings may suggest",
    ),
]

_INPUT_REFUSAL = (
    "Posso responder apenas perguntas médicas relacionadas a oncologia e diagnóstico. "
    "Por favor, reformule sua pergunta dentro desse escopo."
)

_PRESCRIPTION_REFUSAL = (
    "Não posso prescrever medicamentos ou tratamentos diretamente. "
    "Por favor, consulte um médico qualificado para orientação de tratamento."
)

_INSUFFICIENT_CONTEXT = (
    "Não há contexto suficiente nos registros do paciente e na base de conhecimento "
    "para responder com segurança."
)


def filter_input(query: str) -> tuple[bool, str | None]:
    """Returns (is_blocked, refusal_message | None).

    Blocks empty/whitespace queries, obvious off-topic patterns, and
    prescription requests. Medical queries pass through.
    """
    if not query or not query.strip():
        return True, _INPUT_REFUSAL

    if _PRESCRIPTION_PATTERNS.search(query):
        return True, _PRESCRIPTION_REFUSAL

    if _OFF_TOPIC_PATTERNS.search(query):
        return True, _INPUT_REFUSAL

    return False, None


def has_definitive_diagnosis(response: str) -> bool:
    """Return True if response contains definitive diagnosis phrasing that filter_output would soften."""
    if not isinstance(response, str):
        return False
    return any(pattern.search(response) for pattern, _ in _DEFINITIVE_DIAGNOSIS_PATTERNS)


def _soften_treatment_language(response: str) -> str:
    if _TREATMENT_PATTERNS.search(response):
        return _PRESCRIPTION_REFUSAL
    return response


def has_guardrail_signal(original_response: str, filtered_response: str) -> bool:
    if not isinstance(original_response, str) or not isinstance(filtered_response, str):
        return True
    return original_response.strip() != filtered_response.strip()


def filter_output(response: str, *, has_context: bool) -> str:
    """Return a guarded response with insufficiency handling and one disclaimer."""
    if not isinstance(response, str):
        return _DISCLAIMER.lstrip()

    if not has_context:
        response = _INSUFFICIENT_CONTEXT

    for pattern, replacement in _DEFINITIVE_DIAGNOSIS_PATTERNS:
        response = pattern.sub(replacement, response)

    response = _soften_treatment_language(response)

    if _DISCLAIMER not in response:
        response += _DISCLAIMER

    return response
