from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CaseExplanation:
    summary_for_clinician: list[str]
    key_factors: list[str]
    recommended_next_steps: list[str]
    limitations_and_caveats: list[str]
    confidence_statement: str


@dataclass
class ExperimentSummary:
    summary: str
    key_improvements: list[str]
    tradeoffs: list[str]
    limitations: list[str]


def parse_qa_pairs(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array of Q&A dicts from an LLM response.

    Handles markdown code fences (```json ... ```) and bare arrays.
    Raises ValueError if no valid array is found.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM response")
    return json.loads(match.group(0))


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(value).strip()]


def parse_case_explanation(text: str) -> CaseExplanation | None:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return None
    summary = _ensure_list(payload.get("summary_for_clinician"))
    key_factors = _ensure_list(payload.get("key_factors"))
    steps = _ensure_list(payload.get("recommended_next_steps"))
    limits = _ensure_list(payload.get("limitations_and_caveats"))
    confidence = str(payload.get("confidence_statement", "")).strip()
    if not summary or not confidence:
        return None
    return CaseExplanation(
        summary_for_clinician=summary,
        key_factors=key_factors,
        recommended_next_steps=steps,
        limitations_and_caveats=limits,
        confidence_statement=confidence,
    )


def parse_experiment_summary(text: str) -> ExperimentSummary | None:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        return None
    summary = str(payload.get("summary", "")).strip()
    key_improvements = _ensure_list(payload.get("key_improvements"))
    tradeoffs = _ensure_list(payload.get("tradeoffs"))
    limitations = _ensure_list(payload.get("limitations"))
    if not summary:
        return None
    return ExperimentSummary(
        summary=summary,
        key_improvements=key_improvements,
        tradeoffs=tradeoffs,
        limitations=limitations,
    )
