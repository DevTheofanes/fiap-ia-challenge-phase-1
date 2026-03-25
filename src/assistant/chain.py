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
    """Build the final generation prompt for the local assistant model."""
    return (
        "You are a medical oncology assistant helping doctors.\n"
        "Use the provided Patient Context and Knowledge Base Context only.\n"
        "Do not prescribe medication.\n"
        "Do not make definitive diagnoses.\n"
        "If information is insufficient, say so clearly.\n"
        "Answer in the same language as the question.\n"
        "Do not fabricate sources.\n\n"
        "Patient Context\n"
        f"{patient_context}\n\n"
        "Knowledge Base Context\n"
        f"{kb_context}\n\n"
        "Question\n"
        f"{question}\n\n"
        "Answer:"
    )
