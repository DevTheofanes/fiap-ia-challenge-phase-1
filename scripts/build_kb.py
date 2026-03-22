#!/usr/bin/env python
"""Build the medical knowledge base from MedQuAD for M3 RAG.

Downloads MedQuAD from GitHub (if not present), filters cancer/oncology Q&A,
writes .txt files to data/kb/, and builds the ChromaDB vectorstore.

Usage:
    python scripts/build_kb.py
"""
from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET

from src.assistant.retriever import build_vectorstore
from src.config import KB_DIR, LOGS_DIR, MEDQUAD_DIR, VECTORSTORE_DIR
from src.logging_utils import log_event, setup_json_logger

_MEDQUAD_REPO = "https://github.com/abachaa/MedQuAD.git"
# MEDQUAD_DIR = data/external/medquad/1_CancerGov_QA
# _MEDQUAD_ROOT = data/external/medquad  (the clone target)
_MEDQUAD_ROOT = MEDQUAD_DIR.parent

logger = setup_json_logger("build_kb", LOGS_DIR / "build_kb.jsonl")


def _clone_medquad() -> None:
    """Clone MedQuAD repo if not already present."""
    if (_MEDQUAD_ROOT / ".git").exists():
        print(f"MedQuAD already at {_MEDQUAD_ROOT} — skipping clone.")
        return
    print(f"Cloning MedQuAD to {_MEDQUAD_ROOT} …")
    _MEDQUAD_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", _MEDQUAD_REPO, str(_MEDQUAD_ROOT)],
        check=True,
    )
    print("Clone complete.")


def _parse_xml_file(xml_path: Path) -> list[dict]:
    """Parse a MedQuAD XML file into a list of {question, answer} dicts."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    pairs = []
    for qa in root.iter("QAPair"):
        q_el = qa.find("Question")
        a_el = qa.find("Answer")
        if q_el is None or a_el is None:
            continue
        question = (q_el.text or "").strip()
        answer = (a_el.text or "").strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def build_kb() -> int:
    """Parse MedQuAD cancer Q&A, write to KB_DIR (clears existing .txt files first). Returns count of files written."""
    if not MEDQUAD_DIR.exists():
        print(f"ERROR: MedQuAD source not found at {MEDQUAD_DIR}", file=sys.stderr)
        print("Run with a network connection so the repo can be cloned.", file=sys.stderr)
        sys.exit(1)

    if KB_DIR.exists():
        for f in KB_DIR.glob("*.txt"):
            f.unlink()
    KB_DIR.mkdir(parents=True, exist_ok=True)
    xml_files = list(MEDQUAD_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {MEDQUAD_DIR}")

    count = 0
    for xml_path in xml_files:
        pairs = _parse_xml_file(xml_path)
        for i, pair in enumerate(pairs):
            out_path = KB_DIR / f"{xml_path.stem}_{i:04d}.txt"
            out_path.write_text(
                f"Q: {pair['question']}\nA: {pair['answer']}\n", encoding="utf-8"
            )
            count += 1

    print(f"Written {count} Q&A files to {KB_DIR}")
    log_event(logger, "build_kb", files_written=count, kb_dir=str(KB_DIR))
    return count


def main() -> None:
    _clone_medquad()

    count = build_kb()
    if count == 0:
        print("ERROR: No Q&A pairs extracted. Check MedQuAD source files.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBuilding ChromaDB vectorstore at {VECTORSTORE_DIR} …")
    build_vectorstore(KB_DIR, VECTORSTORE_DIR)
    print("Vectorstore built successfully.")
    log_event(logger, "build_vectorstore", vectorstore_dir=str(VECTORSTORE_DIR))


if __name__ == "__main__":
    main()
