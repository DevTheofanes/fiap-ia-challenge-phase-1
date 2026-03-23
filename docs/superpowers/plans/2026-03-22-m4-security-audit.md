# M4 — Security, Explainability & Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement guardrails, audit logging, and explainability for the medical assistant by filling three stub modules and refactoring three graph node functions to delegate to them.

**Architecture:** Three pure-function modules (`guardrails.py`, `audit_logger.py`, `explainer.py`) are implemented first, then `graph.py` is updated to call them at the appropriate nodes. The graph topology and `AssistantState` schema remain unchanged. Each module is independently testable.

**Tech Stack:** Python 3.11+, `langchain-google-genai` (Gemini), `src/logging_utils.py` (`setup_json_logger` + `log_event`), `re` (stdlib regex for guardrails)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/assistant/guardrails.py` | Implement (was stub) | Input/output safety filters |
| `src/assistant/audit_logger.py` | Implement (was stub) | Write structured records to `audit.jsonl` |
| `src/assistant/explainer.py` | Implement (was stub) | Gemini-based citation explanation |
| `src/assistant/graph.py` | Modify 3 node functions | Delegate to M4 modules |
| `tests/test_guardrails.py` | Create | Unit tests for filter functions |
| `tests/test_audit_logger.py` | Create | Unit test for log_interaction output |

---

## Task 1: Implement `guardrails.py`

**Files:**
- Modify: `src/assistant/guardrails.py`
- Create: `tests/test_guardrails.py`

### Background

`filter_input(query)` is a cheap regex pre-filter that runs *before* the LLM classification call in `classify_intent`. It blocks empty queries, obvious off-topic patterns, and prescription requests. `filter_output(response)` owns disclaimer injection (the existing `+ _DISCLAIMER` inline concat in `graph.py` will be removed in Task 4) and softens definitive diagnosis phrasing.

The disclaimer constant `_DISCLAIMER` is currently defined in `graph.py`. `guardrails.py` will define its own copy of the same string since it owns disclaimer injection going forward.

- [ ] **Step 1: Write failing tests**

Create `tests/test_guardrails.py`:

```python
import pytest
from src.assistant.guardrails import filter_input, filter_output

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

_DISCLAIMER = "\n\n⚠ Esta informação é educacional e não substitui consulta médica profissional."

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

def test_filter_output_preserves_normal_response():
    response = "Breast cancer symptoms may include a lump or skin changes."
    result = filter_output(response)
    assert "Breast cancer symptoms may include" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/theonetto/www/pos-ia-dev/fiap-ia-challenge-phase-2
source .venv/bin/activate
python -m pytest tests/test_guardrails.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — functions not yet implemented.

- [ ] **Step 3: Implement `src/assistant/guardrails.py`**

```python
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
    r"crypto|recipe|receita culin|movie|filme|music|m[uú]sica|política|política|politics|"
    r"lottery|loteria|horoscope|hor[oó]scopo)\b",
    re.IGNORECASE,
)

_PRESCRIPTION_PATTERNS = re.compile(
    r"\b(prescri(be|va|va-me)|give me (medication|medicine|drug|pills?)|"
    r"me d[êe] (rem[eé]dio|medica[çc][aã]o)|quero (comprar|tomar) (rem[eé]dio|medica[çc][aã]o))\b",
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


def filter_output(response: str) -> str:
    """Appends disclaimer if absent; softens definitive diagnosis phrasing."""
    for pattern, replacement in _DEFINITIVE_DIAGNOSIS_PATTERNS:
        response = pattern.sub(replacement, response)

    if _DISCLAIMER not in response:
        response += _DISCLAIMER

    return response
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_guardrails.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/assistant/guardrails.py tests/test_guardrails.py
git commit -m "feat(m4): implement guardrails filter_input and filter_output"
```

---

## Task 2: Implement `audit_logger.py`

**Files:**
- Modify: `src/assistant/audit_logger.py`
- Create: `tests/test_audit_logger.py`

### Background

Uses `setup_json_logger("audit", AUDIT_LOG_PATH)` + `log_event` from `src/logging_utils.py`. `AUDIT_LOG_PATH` is already defined in `src/config.py` as `artifacts/logs/audit.jsonl`. The logger is a module-level singleton initialized on first call to avoid re-adding file handlers.

- [ ] **Step 1: Write failing test**

Create `tests/test_audit_logger.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.assistant.audit_logger import log_interaction


def test_log_interaction_writes_required_fields(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        # Reset the module-level logger so it picks up the patched path
        import src.assistant.audit_logger as mod
        mod._logger = None

        log_interaction(
            user_query="What is oncology?",
            retrieved_docs=["medquad.txt"],
            model_response="Oncology is the study of cancer.",
            guardrail_triggered=False,
            intent="medical",
            error=None,
        )

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert len(records) == 1
    rec = records[0]
    assert "timestamp" in rec
    assert rec["user_query"] == "What is oncology?"
    assert rec["retrieved_docs"] == ["medquad.txt"]
    assert rec["model_response"] == "Oncology is the study of cancer."
    assert rec["guardrail_triggered"] is False
    assert rec["intent"] == "medical"


def test_log_interaction_guardrail_triggered(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    with patch("src.assistant.audit_logger.AUDIT_LOG_PATH", audit_path):
        import src.assistant.audit_logger as mod
        mod._logger = None

        log_interaction(
            user_query="What's the weather?",
            retrieved_docs=[],
            model_response="Out of scope.",
            guardrail_triggered=True,
        )

    records = [json.loads(line) for line in audit_path.read_text().splitlines() if line]
    assert records[0]["guardrail_triggered"] is True
    assert records[0]["retrieved_docs"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_audit_logger.py -v 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `src/assistant/audit_logger.py`**

```python
"""Audit logger for medical assistant interactions (M4).

Logs all interactions to artifacts/logs/audit.jsonl with:
timestamp, user_query, retrieved_docs, model_response, guardrail_triggered.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import AUDIT_LOG_PATH
from src.logging_utils import log_event, setup_json_logger

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_json_logger("audit", AUDIT_LOG_PATH)
    return _logger


def log_interaction(
    *,
    user_query: str,
    retrieved_docs: list[str],
    model_response: str,
    guardrail_triggered: bool,
    intent: str | None = None,
    error: str | None = None,
) -> None:
    """Appends one structured record to artifacts/logs/audit.jsonl.

    Required fields: timestamp, user_query, retrieved_docs,
                     model_response, guardrail_triggered.
    Optional fields: intent, error.
    """
    log_event(
        _get_logger(),
        "audit",
        user_query=user_query,
        retrieved_docs=retrieved_docs,
        model_response=model_response,
        guardrail_triggered=guardrail_triggered,
        intent=intent,
        error=error,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_audit_logger.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/assistant/audit_logger.py tests/test_audit_logger.py
git commit -m "feat(m4): implement audit_logger log_interaction"
```

---

## Task 3: Implement `explainer.py`

**Files:**
- Modify: `src/assistant/explainer.py`
- Test: manual smoke test (no unit test — Gemini is mocked via env var)

### Background

`explain_prediction` sends a prompt to Gemini via `ChatGoogleGenerativeAI` (same pattern as `graph.py`). When `LLM_USE_MOCK=true` or the API call fails, it falls back to the template string `"Based on: {sources_joined}."`. Returns empty string when `sources` is empty.

- [ ] **Step 1: Implement `src/assistant/explainer.py`**

```python
"""Explainer wrapper for model prediction explanations (M4).

Uses Gemini to produce a citation-based 1-2 sentence explanation of why
the assistant gave a particular response.
"""
from __future__ import annotations

import os

_PROMPT_TEMPLATE = """\
You are assisting a doctor reviewing a medical assistant's response.
In 1-2 sentences, explain why the assistant gave this answer, \
citing the specific source(s) it used.

Question: {query}
Sources consulted: {sources}
{ml_context_block}
Assistant response (excerpt): {response_excerpt}

Reply in the same language as the question. Be concise."""


def explain_prediction(
    *,
    query: str,
    response: str,
    sources: list[str],
    ml_context: str | None = None,
) -> str:
    """Uses Gemini to produce a 1-2 sentence explanation citing sources.

    Returns empty string if sources list is empty.
    Falls back to template string on API error or when LLM_USE_MOCK=true.
    """
    if not sources:
        return ""

    sources_joined = ", ".join(sources)
    fallback = f"Based on: {sources_joined}."

    if os.getenv("LLM_USE_MOCK", "false").lower() == "true":
        return fallback

    ml_context_block = f"ML Pipeline Context: {ml_context}\n" if ml_context else ""
    response_excerpt = response[:300]

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        sources=sources_joined,
        ml_context_block=ml_context_block,
        response_excerpt=response_excerpt,
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception:
        return fallback
```

- [ ] **Step 2: Smoke test with mock mode**

```bash
LLM_USE_MOCK=true python -c "
from src.assistant.explainer import explain_prediction
result = explain_prediction(
    query='What are symptoms of breast cancer?',
    response='Common symptoms include a lump.',
    sources=['medquad_oncology.txt'],
    ml_context=None,
)
print('Result:', repr(result))
assert result == 'Based on: medquad_oncology.txt.', f'Unexpected: {result}'
print('PASS: mock fallback correct')
"
```

Expected output: `PASS: mock fallback correct`

- [ ] **Step 3: Smoke test with empty sources**

```bash
LLM_USE_MOCK=true python -c "
from src.assistant.explainer import explain_prediction
result = explain_prediction(
    query='test', response='test', sources=[], ml_context=None
)
assert result == '', f'Expected empty string, got: {result!r}'
print('PASS: empty sources returns empty string')
"
```

Expected output: `PASS: empty sources returns empty string`

- [ ] **Step 4: Commit**

```bash
git add src/assistant/explainer.py
git commit -m "feat(m4): implement explainer explain_prediction with Gemini and fallback"
```

---

## Task 4: Update `graph.py` — delegate to M4 modules

**Files:**
- Modify: `src/assistant/graph.py`

### Background

Three node functions are updated. The graph topology (nodes, edges) and `AssistantState` are unchanged. The key changes:

1. `classify_intent`: call `filter_input` before the LLM; if blocked, set `intent="out_of_scope"` and `response=refusal_message`, skip LLM.
2. `validate_response`: remove the inline `+ _DISCLAIMER`; call `filter_output`, then `explain_prediction`, then `log_interaction`.
3. `refuse_response`: use `state.get("response") or _REFUSAL_MSG` (preserves guardrail message); call `log_interaction`.

- [ ] **Step 1: Read current `graph.py` to confirm structure**

Read `src/assistant/graph.py` lines 1–145 before editing.

- [ ] **Step 2: Add M4 imports and `DISCLAIMER` constant to `graph.py`**

Add at the top-level import block (alongside the existing `from src.assistant.chain import build_chain` etc.):

```python
from src.assistant import audit_logger, explainer, guardrails
from src.assistant.guardrails import _DISCLAIMER
```

`_DISCLAIMER` is imported from `guardrails.py` so that `validate_response` always uses the same string that `filter_output` inserts — a single source of truth. The existing `_DISCLAIMER` constant defined locally in `graph.py` can be removed after this import is added.

- [ ] **Step 3: Update `classify_intent` node**

Replace the body of `classify_intent` with:

```python
    def classify_intent(state: AssistantState) -> AssistantState:
        is_blocked, refusal_message = guardrails.filter_input(state["query"])
        if is_blocked:
            return {**state, "intent": "out_of_scope", "response": refusal_message, "error": None}
        try:
            prompt = _CLASSIFY_PROMPT.format(query=state["query"])
            result = llm.invoke(prompt)
            text = result.content.strip().lower()
            intent = "medical" if text == "medical" or text.startswith("medical") else "out_of_scope"
        except Exception as exc:
            intent = "out_of_scope"
            return {**state, "intent": intent, "error": str(exc)}
        return {**state, "intent": intent, "error": None}
```

- [ ] **Step 4: Update `validate_response` node**

Replace the body of `validate_response` with:

```python
    def validate_response(state: AssistantState) -> AssistantState:
        sources = [
            Path(doc.metadata["source"]).name if "source" in doc.metadata else "unknown"
            for doc in state.get("retrieved_docs", [])
        ]
        response = guardrails.filter_output(state["response"])
        explanation = explainer.explain_prediction(
            query=state["query"],
            response=response,
            sources=sources,
            ml_context=state["ml_context"],
        )
        if explanation:
            # _DISCLAIMER is imported from guardrails.py — single source of truth
            response = response.replace(
                _DISCLAIMER,
                f"\n\nExplanation: {explanation}{_DISCLAIMER}",
            )
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=sources,
            model_response=response,
            guardrail_triggered=False,
            intent=state["intent"],
            error=state.get("error"),
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            sources=sources,
            response_len=len(response),
            refused=False,
        )
        return {**state, "response": response, "sources": sources, "refused": False}
```

Note: `_DISCLAIMER` is imported from `guardrails.py` (Step 2 above), so this `str.replace` always uses the same string that `filter_output` inserts.

- [ ] **Step 5: Update `refuse_response` node**

Replace the body of `refuse_response` with:

```python
    def refuse_response(state: AssistantState) -> AssistantState:
        final_response = state.get("response") or _REFUSAL_MSG
        audit_logger.log_interaction(
            user_query=state["query"],
            retrieved_docs=[],
            model_response=final_response,
            guardrail_triggered=True,
            intent=state["intent"],
            error=state.get("error"),
        )
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            sources=[],
            response_len=len(final_response),
            refused=True,
            error=state.get("error"),
        )
        return {**state, "response": final_response, "sources": [], "refused": True}
```

- [ ] **Step 6: End-to-end smoke test — medical query**

```bash
LLM_USE_MOCK=false python scripts/run_assistant.py
```

Type: `What are the symptoms of breast cancer?`

Expected:
- Response body with answer
- `Explanation: Based on: <source>.` line (or Gemini-generated)
- Disclaimer line at end
- `Sources: <filename>` printed by CLI

Then check audit log:
```bash
python -c "
import json
from pathlib import Path
lines = Path('artifacts/logs/audit.jsonl').read_text().splitlines()
last = json.loads(lines[-1])
assert 'user_query' in last
assert 'guardrail_triggered' in last
assert last['guardrail_triggered'] is False
print('PASS audit.jsonl fields present:', list(last.keys()))
"
```

- [ ] **Step 7: End-to-end smoke test — guardrail triggers**

In the running assistant, type: `What is the weather today?`

Expected: refusal message (not an LLM call), `guardrail_triggered: true` in audit log.

Then verify:
```bash
python -c "
import json
from pathlib import Path
lines = Path('artifacts/logs/audit.jsonl').read_text().splitlines()
last = json.loads(lines[-1])
assert last['guardrail_triggered'] is True
print('PASS guardrail_triggered=True recorded')
"
```

- [ ] **Step 8: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/assistant/graph.py
git commit -m "feat(m4): integrate guardrails, audit_logger, explainer into graph nodes"
```

---

## Summary

After all four tasks:

| Done | Deliverable |
|------|-------------|
| ✅ | `guardrails.filter_input` blocks empty/off-topic/prescription queries |
| ✅ | `guardrails.filter_output` owns disclaimer injection + diagnosis softening |
| ✅ | `audit_logger.log_interaction` writes to `artifacts/logs/audit.jsonl` |
| ✅ | `explainer.explain_prediction` cites sources via Gemini with fallback |
| ✅ | `graph.py` delegates to all three modules; topology unchanged |
| ✅ | Every interaction (refused or not) produces an audit record |
