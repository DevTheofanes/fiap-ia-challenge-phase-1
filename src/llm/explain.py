from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import LLMClient, get_llm_client
from .prompts import format_case_prompt
from .schemas import CaseExplanation, parse_case_explanation


@dataclass
class CasePayload:
    diagnosis: str
    probability: float | None
    threshold: float | None
    threshold_reason: str
    evidence: list[dict[str, Any]]
    safety_notes: list[str]


def _default_safety_notes() -> list[str]:
    return [
        "This output does not replace clinical judgment.",
        "Model trained on a public dataset; external validity may vary.",
        "Potential bias or data drift may affect predictions.",
    ]


def build_case_payload(
    *,
    diagnosis: str,
    probability: float | None,
    threshold: float | None,
    threshold_reason: str,
    top_features: list[dict[str, Any]],
    safety_notes: list[str] | None = None,
) -> CasePayload:
    return CasePayload(
        diagnosis=diagnosis,
        probability=probability,
        threshold=threshold,
        threshold_reason=threshold_reason,
        evidence=top_features,
        safety_notes=safety_notes or _default_safety_notes(),
    )


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scrubbed = json.loads(json.dumps(payload))
    for feature in scrubbed.get("evidence", []):
        if "value" in feature:
            feature["value"] = "[redacted]"
    return scrubbed


def _log_interaction(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _template_explanation(payload: CasePayload) -> CaseExplanation:
    prob_text = "" if payload.probability is None else f"Estimated probability: {payload.probability:.2f}."
    threshold_text = (
        "" if payload.threshold is None else f"Decision threshold: {payload.threshold:.2f} ({payload.threshold_reason})."
    )
    factors = [item.get("feature", "unknown") for item in payload.evidence]
    summary = [
        f"Model predicts {payload.diagnosis}.",
        prob_text,
        threshold_text,
    ]
    summary = [line for line in summary if line]
    return CaseExplanation(
        summary_for_clinician=summary,
        key_factors=factors[:5],
        recommended_next_steps=[
            "Review the exam and correlate with clinical history.",
            "Consider additional investigation if clinically indicated.",
        ],
        limitations_and_caveats=payload.safety_notes,
        confidence_statement="This is a statistical estimate; uncertainty remains.",
    )


def explain_case(
    payload: CasePayload,
    *,
    client: LLMClient | None = None,
    log_path: Path | None = None,
) -> CaseExplanation:
    prompt = format_case_prompt(asdict(payload))
    llm_client = client or get_llm_client()
    used_llm = llm_client is not None
    response_text = None
    explanation: CaseExplanation | None = None

    if used_llm:
        try:
            response = llm_client.generate(prompt)
            response_text = response.text
            explanation = parse_case_explanation(response.text)
        except Exception:
            explanation = None

    if explanation is None:
        explanation = _template_explanation(payload)

    if log_path is not None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "used_llm": used_llm,
            "prompt": prompt,
            "payload": _redact_payload(asdict(payload)),
            "response_text": response_text,
            "final": asdict(explanation),
        }
        _log_interaction(log_path, log_entry)

    return explanation
