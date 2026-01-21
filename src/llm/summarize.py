from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .client import LLMClient, get_llm_client
from .prompts import format_metrics_prompt
from .schemas import ExperimentSummary, parse_experiment_summary


def _log_interaction(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _template_summary() -> ExperimentSummary:
    return ExperimentSummary(
        summary="GA results reviewed. Use the compare table to list metric deltas and trade-offs.",
        key_improvements=[],
        tradeoffs=[],
        limitations=["Automated summary not available; review metrics manually."],
    )


def summarize_experiment(
    table_markdown: str,
    *,
    client: LLMClient | None = None,
    log_path: Path | None = None,
) -> ExperimentSummary:
    prompt = format_metrics_prompt(table_markdown)
    llm_client = client or get_llm_client()
    used_llm = llm_client is not None
    response_text = None
    summary: ExperimentSummary | None = None

    if used_llm:
        try:
            response = llm_client.generate(prompt)
            response_text = response.text
            summary = parse_experiment_summary(response.text)
        except Exception:
            summary = None

    if summary is None:
        summary = _template_summary()

    if log_path is not None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "used_llm": used_llm,
            "prompt": prompt,
            "response_text": response_text,
            "final": asdict(summary),
        }
        _log_interaction(log_path, log_entry)

    return summary
