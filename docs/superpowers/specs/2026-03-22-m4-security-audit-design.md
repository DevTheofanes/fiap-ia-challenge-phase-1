# M4 — Security, Explainability & Audit Design

**Date:** 2026-03-22
**Milestone:** M4 — Segurança, Explicabilidade e Auditoria
**Status:** Approved

---

## Context

M3 is complete. `src/assistant/graph.py` implements a LangGraph StateGraph with 5 nodes:
`classify_intent → retrieve_context → generate_response → validate_response`
and an `out_of_scope` branch to `refuse_response`.

M4 fills the three empty stub files (`guardrails.py`, `audit_logger.py`, `explainer.py`) and
lightly refactors `graph.py` to delegate to them. Graph topology and `AssistantState` are
unchanged. `run_assistant.py` is untouched.

---

## Chosen Approach: Thin Delegation Layer

The three M4 modules become pure functions. `graph.py` node internals call them at the
appropriate points. Each module is self-contained and independently testable.

Rejected alternatives:
- **Post-processing wrapper in CLI**: splits logic across layers; doesn't protect programmatic callers
- **New middleware graph nodes**: adds graph complexity; duplicates the existing classify→refuse guard

---

## Module Interfaces

### `src/assistant/guardrails.py`

```python
def filter_input(query: str) -> tuple[bool, str | None]:
    """Returns (is_blocked, refusal_message | None).

    Blocks:
    - Empty or whitespace-only queries
    - Obvious off-topic triggers via regex (e.g. weather, sports, finance keywords)
    - Prescription-request patterns ("prescribe me", "give me medication", etc.)

    This is a cheap pre-filter. LLM-based scope classification in classify_intent
    still runs for anything that passes here.
    """

def filter_output(response: str) -> str:
    """Ensures output safety.

    - Appends the mandatory disclaimer if not already present.
    - Softens definitive diagnosis phrasing (regex: "you have cancer",
      "diagnosis is X") → hedged equivalent.

    Returns the sanitised response string.
    """
```

### `src/assistant/audit_logger.py`

```python
def log_interaction(
    *,
    user_query: str,
    retrieved_docs: list[str],   # source filenames, not full content
    model_response: str,
    guardrail_triggered: bool,
    intent: str | None = None,
    error: str | None = None,
) -> None:
    """Appends one structured record to artifacts/logs/audit.jsonl.

    Uses setup_json_logger + log_event from src/logging_utils.py.
    Logger name: "audit"
    Path constant: AUDIT_LOG_PATH from src.config (artifacts/logs/audit.jsonl)
    Logger is module-level singleton (initialized once on first call).

    Required fields in output: timestamp, user_query, retrieved_docs,
    model_response, guardrail_triggered.
    Optional fields: intent, error.
    """
```

### `src/assistant/explainer.py`

The milestone requires a function named `explain_prediction`. The function receives
already-computed assistant state (no ML model re-execution) and wraps Gemini to produce
a citation-based explanation — the name reflects its role as explainer of the assistant's
prediction/response.

```python
def explain_prediction(
    *,
    query: str,
    response: str,
    sources: list[str],
    ml_context: str | None = None,
) -> str:
    """Uses Gemini to produce a 1-2 sentence explanation citing sources.

    Prompt instructs Gemini to explain why the assistant gave this answer,
    citing the specific retrieved sources and any ML prediction context.
    Response excerpt (first 300 chars) is used to avoid token bloat.
    ml_context_block is omitted entirely when ml_context is None.

    Fallback (LLM_USE_MOCK=true or API error):
        "Based on: {sources_joined}."

    Returns empty string if sources list is empty (no explanation appended).
    """
```

---

## Graph Changes (`graph.py`)

Graph topology: **unchanged** (same 5 nodes, same edges).
`AssistantState`: **unchanged**.

### `classify_intent` node

```
+ calls guardrails.filter_input(query) → (is_blocked, refusal_message)
+ if is_blocked:
    sets intent="out_of_scope"
    sets response=refusal_message (overrides _REFUSAL_MSG in refuse_response)
    skips LLM call
  else:
    LLM classification as before
```

If `filter_input` blocks the query, `state["response"]` is set to the `refusal_message`
returned by `filter_input`. The `refuse_response` node then uses `state["response"]` if
already set, otherwise falls back to `_REFUSAL_MSG`.

### `validate_response` node

The existing `response = state["response"] + _DISCLAIMER` line is **removed**.
`filter_output` becomes the sole owner of disclaimer injection.

```
# raw_response is state["response"] — chain output, no disclaimer yet
# Reuse existing sources extraction already present in validate_response:
#   sources = [Path(doc.metadata["source"]).name if "source" in doc.metadata else "unknown"
#              for doc in state.get("retrieved_docs", [])]
response = guardrails.filter_output(raw_response)        # appends disclaimer
explanation = explainer.explain_prediction(
    query=state["query"],
    response=response,
    sources=sources,
    ml_context=state["ml_context"],
)
if explanation:
    # insert explanation between answer body and disclaimer
    response = response.replace(_DISCLAIMER, f"\n\nExplanation: {explanation}{_DISCLAIMER}")
audit_logger.log_interaction(
    user_query=state["query"],
    retrieved_docs=sources,          # list[str] filenames, not Document objects
    model_response=response,
    guardrail_triggered=False,
    intent=state["intent"],
    error=state.get("error"),
)
existing log_event() call kept as-is
```

### `refuse_response` node

The existing `return {**state, "response": _REFUSAL_MSG, ...}` line is updated to use
`state.get("response") or _REFUSAL_MSG` so that when `classify_intent` placed a
guardrail-specific message into `state["response"]`, it is preserved.

```
final_response = state.get("response") or _REFUSAL_MSG
audit_logger.log_interaction(
    user_query=state["query"],
    retrieved_docs=[],
    model_response=final_response,
    guardrail_triggered=True,
    intent=state["intent"],
    error=state.get("error"),
)
existing log_event() call kept as-is
return {**state, "response": final_response, "sources": [], "refused": True}
```

---

## Output Format

### Response structure (successful query)

```
{answer body}

Explanation: {1-2 sentences citing sources, from explainer.py}

⚠ Esta informação é educacional e não substitui consulta médica profissional.
```

If `explain_prediction()` returns empty string, the Explanation block is omitted.

### `audit.jsonl` record shape

```json
{
  "timestamp": "2026-03-22T14:00:00Z",
  "level": "INFO",
  "message": "audit",
  "stage": "audit",
  "user_query": "What are the symptoms of breast cancer?",
  "retrieved_docs": ["medquad_oncology.txt"],
  "model_response": "Symptoms include... ⚠ Esta informação...",
  "guardrail_triggered": false,
  "intent": "medical",
  "error": null
}
```

Uses `JsonlFormatter` from `logging_utils.py` — consistent with all other project logs.

---

## Explainer Prompt

```
You are assisting a doctor reviewing a medical assistant's response.
In 1-2 sentences, explain why the assistant gave this answer,
citing the specific source(s) it used.

Question: {query}
Sources consulted: {sources}
{ml_context_block}
Assistant response (excerpt): {response_excerpt}

Reply in the same language as the question. Be concise.
```

Fallback string: `"Based on: {sources_joined}."`

---

## Files Modified

| File | Change |
|------|--------|
| `src/assistant/guardrails.py` | Implement `filter_input()`, `filter_output()` |
| `src/assistant/audit_logger.py` | Implement `log_interaction()` |
| `src/assistant/explainer.py` | Implement `explain_prediction()` |
| `src/assistant/graph.py` | Delegate to M4 modules in 3 node functions |

No new files. No new graph nodes. No new CLI commands.

---

## Out of Scope

- Changes to `run_assistant.py`
- Changes to `AssistantState` schema
- New LangGraph nodes
- ML model re-execution inside `explainer.py` (uses pre-computed `ml_context` string)
- Replacing existing `assistant.jsonl` logging (audit.jsonl is additive)
