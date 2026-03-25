"""Prompt and context helpers for the medical assistant."""
from __future__ import annotations

from langchain_core.documents import Document


def format_kb_context(docs: list[Document]) -> str:
    """Render KB excerpts into a prompt-friendly block."""
    if not docs:
        return "No relevant knowledge base context found."

    blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[KB-{idx}] source={source}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_generation_prompt(
    *,
    question: str,
    patient_context: str,
    kb_context: str,
) -> str:
    """Build the generation prompt using the same format as fine-tuning training data."""
    context_parts: list[str] = []

    _no_patient = "No patient-specific context provided."
    if patient_context and patient_context.strip() and patient_context.strip() != _no_patient:
        context_parts.append(f"Patient information:\n{patient_context.strip()}")

    _no_kb = "No relevant knowledge base context found."
    if kb_context and kb_context.strip() and kb_context.strip() != _no_kb:
        context_parts.append(f"Medical reference:\n{kb_context.strip()}")

    if context_parts:
        context_block = "\n\n".join(context_parts)
        return f"### Question: {question}\n\nContext:\n{context_block}\n\n### Answer:"
    return f"### Question: {question}\n\n### Answer:"
