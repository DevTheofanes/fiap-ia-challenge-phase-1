"""M1 — Prepare fine-tuning data (instruction pairs) for oncology LLM.

Generates synthetic Q&A pairs via Gemini and optionally merges with:
  --input-file   : generic JSONL with {"question":..., "answer":...} keys
  --pubmedqa-file: PubMedQA ori_pqal.json (dict keyed by PMID with
                   QUESTION / LONG_ANSWER / CONTEXTS / MESHES fields)

Outputs train/val splits to data/finetune/train.jsonl and val.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config as config
from src.llm.client import get_llm_client
from src.llm.prompts import format_synthetic_qa_prompt
from src.llm.schemas import parse_qa_pairs
from src.logging_utils import log_event, setup_json_logger

MIN_TRAIN_RECORDS = 500

# ---------------------------------------------------------------------------
# Keywords used to filter oncology-relevant rows
# ---------------------------------------------------------------------------
_ONCOLOGY_KEYWORDS = {
    "cancer", "tumor", "tumour", "carcinoma", "biopsy", "malignant",
    "malignancy", "oncology", "oncologist", "breast", "mammogram",
    "metastasis", "metastatic", "chemotherapy", "radiation", "radiotherapy",
    "staging", "prognosis", "pathology", "biomarker", "hormone receptor",
    "her2", "estrogen", "progesterone", "lumpectomy", "mastectomy",
    "lymph node", "benign", "lesion", "adenocarcinoma", "sarcoma",
}

# ---------------------------------------------------------------------------
# Anonymization — ordered patterns applied left-to-right
# ---------------------------------------------------------------------------
import re as _re

_ANON_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    # ISO dates: 2023-07-14
    (_re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    # US dates: 07/14/2023 or 7/14/23
    (_re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
    # Written dates: January 14, 2023 or Jan 14 2023
    (_re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\s+\d{1,2}[,\s]+\d{4}\b",
        _re.IGNORECASE,
    ), "[DATE]"),
    # Age expressions: 55-year-old, age 42, aged 70
    (_re.compile(r"\b(\d{1,3})[- ]?year[- ]?old\b", _re.IGNORECASE), "[AGE]-year-old"),
    (_re.compile(r"\bage[d]?\s+\d{1,3}\b", _re.IGNORECASE), "age [AGE]"),
    # Patient IDs: MRN 123456, ID: 7890
    (_re.compile(r"\b(?:MRN|patient\s*id|pt\s*id|ID)[:\s#]*\d+\b", _re.IGNORECASE), "[PATIENT_ID]"),
    # 5+ digit numbers (potential IDs)
    (_re.compile(r"\b\d{5,}\b"), "[NUM]"),
    # "Dr. Firstname Lastname" or "Dr. Lastname"
    (_re.compile(r"\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"), "Dr. [NAME]"),
    # "patient <Name>" context — title-case name after "patient" (case-insensitive)
    (_re.compile(r"\bpatient\s+[A-Z][a-z]+\b", _re.IGNORECASE), "patient [NAME]"),
    # Institution names: capitalized multi-word ending in Hospital/Center/Clinic/Institute
    (_re.compile(
        r"\b(?:[A-Z][a-z]+\s+){1,4}(?:Hospital|Medical Center|Cancer Center|Clinic|Institute)\b"
    ), "[INSTITUTION]"),
    # Phone numbers
    (_re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    # Email addresses
    (_re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), "[EMAIL]"),
]


def _anonymize(text: str) -> str:
    for pattern, replacement in _ANON_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _is_oncology_relevant(text: str) -> bool:
    return any(kw in text for kw in _ONCOLOGY_KEYWORDS)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_pubmedqa_json(path: Path, logger: Any) -> list[dict[str, str]]:
    """Load PubMedQA ori_pqal.json (dict keyed by PMID).

    Each record has:
      QUESTION    — the research question
      LONG_ANSWER — the long-form answer from the abstract conclusion
      CONTEXTS    — list of abstract paragraph strings (fallback if LONG_ANSWER empty)
      MESHES      — MeSH terms (included in oncology filter check)

    Keeps records whose combined text contains an oncology keyword.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    pairs: list[dict[str, str]] = []
    skipped = 0

    for _pmid, record in raw.items():
        question = str(record.get("QUESTION", "")).strip()
        if not question:
            skipped += 1
            continue

        answer = str(record.get("LONG_ANSWER", "")).strip()
        if not answer:
            contexts = record.get("CONTEXTS", [])
            answer = " ".join(str(c) for c in contexts).strip()

        if not answer:
            skipped += 1
            continue

        meshes = " ".join(record.get("MESHES", [])).lower()
        combined = (question + " " + answer + " " + meshes).lower()
        if _is_oncology_relevant(combined):
            pairs.append({"question": question, "answer": answer})
        else:
            skipped += 1

    log_event(logger, "pubmedqa_loaded", path=str(path), total=len(raw), kept=len(pairs), skipped=skipped)
    return pairs


def _load_input_file(path: Path, logger: Any) -> list[dict[str, str]]:
    """Load a generic JSONL file and filter for oncology-relevant rows."""
    pairs: list[dict[str, str]] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                skipped += 1
                continue
            if _is_oncology_relevant((question + " " + answer).lower()):
                pairs.append({"question": question, "answer": answer})
            else:
                skipped += 1
    log_event(logger, "input_file_loaded", path=str(path), kept=len(pairs), skipped=skipped)
    return pairs


def _load_medquad_dir(path: Path, logger: Any) -> list[dict[str, str]]:
    """Load MedQuAD XML files from a directory and filter for oncology-relevant rows."""
    pairs: list[dict[str, str]] = []
    skipped = 0
    xml_files = sorted(path.rglob("*.xml"))

    for xml_file in xml_files:
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            skipped += 1
            continue

        focus = (root.findtext("Focus") or "").strip()
        for qa_pair in root.findall("./QAPairs/QAPair"):
            question = " ".join((qa_pair.findtext("Question") or "").split())
            answer = " ".join((qa_pair.findtext("Answer") or "").split())
            if not question or not answer:
                skipped += 1
                continue

            combined = (focus + " " + question + " " + answer).lower()
            if _is_oncology_relevant(combined):
                pairs.append({"question": question, "answer": answer})
            else:
                skipped += 1

    log_event(logger, "medquad_loaded", path=str(path), files=len(xml_files), kept=len(pairs), skipped=skipped)
    return pairs


def _generate_synthetic(count: int, client: Any, logger: Any) -> list[dict[str, str]]:
    """Generate `count` synthetic Q&A pairs via Gemini in batches of 30.

    Stops when `count` pairs are collected or after `max_attempts` batches,
    whichever comes first.
    """
    batch_size = 30
    max_attempts = count  # upper bound: one item per attempt in the worst case
    pairs: list[dict[str, str]] = []
    batch = 0

    while len(pairs) < count and batch < max_attempts:
        batch += 1
        n = min(batch_size, count - len(pairs))
        prompt = format_synthetic_qa_prompt(n)
        try:
            response = client.generate(prompt, temperature=0.8, max_tokens=8192)
            items = parse_qa_pairs(response.text)
            valid = [
                {"question": str(it["question"]).strip(), "answer": str(it["answer"]).strip()}
                for it in items
                if isinstance(it, dict) and it.get("question") and it.get("answer")
            ]
            pairs.extend(valid)
            log_event(logger, "synthetic_batch_ok", batch=batch, requested=n, received=len(valid), total_so_far=len(pairs))
        except Exception as exc:
            log_event(logger, "synthetic_batch_error", batch=batch, error=str(exc))

    return pairs[:count]


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fine-tuning JSONL dataset")
    parser.add_argument(
        "--total",
        type=int,
        default=625,
        help="Target total pairs before train/val split (default supports train.jsonl >= 500 with val_split=0.2)",
    )
    parser.add_argument("--input-file", type=Path, default=None, help="JSONL with question/answer keys")
    parser.add_argument("--pubmedqa-file", type=Path, default=None, help="PubMedQA ori_pqal.json")
    parser.add_argument("--medquad-dir", type=Path, default=None, help="Directory with MedQuAD XML files")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = parser.parse_args()

    logger = setup_json_logger("finetune_prep", config.FINETUNE_PREP_LOG_PATH)
    log_event(logger, "prepare_start", total=args.total, seed=args.seed,
              input_file=str(args.input_file), pubmedqa_file=str(args.pubmedqa_file))

    all_pairs: list[dict[str, str]] = []

    if args.pubmedqa_file is not None:
        all_pairs.extend(_load_pubmedqa_json(args.pubmedqa_file, logger))

    if args.medquad_dir is not None:
        all_pairs.extend(_load_medquad_dir(args.medquad_dir, logger))

    if args.input_file is not None:
        all_pairs.extend(_load_input_file(args.input_file, logger))

    synthetic_needed = max(0, args.total - len(all_pairs))
    if synthetic_needed > 0:
        client = get_llm_client()
        if client is None:
            print("ERROR: No LLM client available. Set GEMINI_API_KEY or LLM_USE_MOCK=true.", file=sys.stderr)
            sys.exit(1)
        all_pairs.extend(_generate_synthetic(synthetic_needed, client, logger))

    # Deduplicate by lowercased question
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for p in all_pairs:
        key = p["question"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    log_event(logger, "dedup_done", before=len(all_pairs), after=len(deduped))

    min_total_records = math.ceil(MIN_TRAIN_RECORDS / (1 - args.val_split))

    if len(deduped) < min_total_records:
        raise RuntimeError(
            f"Insufficient pairs collected for M1: {len(deduped)} < {min_total_records}. "
            f"Need enough records to keep at least {MIN_TRAIN_RECORDS} rows in train.jsonl "
            f"with val_split={args.val_split}. Check GEMINI_API_KEY or provide --input-file / --pubmedqa-file."
        )

    records = [{"prompt": _anonymize(p["question"]), "completion": _anonymize(p["answer"])} for p in deduped]

    train_records, val_records = train_test_split(records, test_size=args.val_split, random_state=args.seed)

    if len(train_records) < MIN_TRAIN_RECORDS:
        raise RuntimeError(
            f"train.jsonl has {len(train_records)} rows, below the M1 requirement of {MIN_TRAIN_RECORDS}. "
            f"Increase --total or reduce --val-split."
        )

    config.FINETUNE_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.FINETUNE_TRAIN_PATH, train_records)
    _write_jsonl(config.FINETUNE_VAL_PATH, val_records)

    log_event(logger, "prepare_done", train=len(train_records), val=len(val_records), total=len(records),
              train_path=str(config.FINETUNE_TRAIN_PATH), val_path=str(config.FINETUNE_VAL_PATH))
    print(f"Done. train={len(train_records)} val={len(val_records)} total={len(records)} seed={args.seed}")


if __name__ == "__main__":
    main()
