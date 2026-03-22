# M3 — Medical Assistant Design Spec
**Date:** 2026-03-22
**Milestone:** M3 — Assistente Médico com LangChain + LangGraph
**Challenge:** FIAP IADT Tech Challenge Fase 3

---

## Context

This spec covers the implementation of the medical assistant pipeline (M3) for the FIAP Fase 3 challenge. It extends the existing breast cancer ML pipeline (Phases 1+2) with a conversational assistant backed by RAG and a LangGraph orchestration flow.

All stub files already exist in `src/assistant/`. The fine-tuned TinyLlama (M2) is still training — M3 starts with `ChatGoogleGenerativeAI` (LangChain-native Gemini) as the generator and swaps in the fine-tuned adapter once M2 completes.

---

## Architecture Overview

### LangGraph Flow (5 nodes, conditional routing)

```
         ┌─────────────────┐
         │  classify_intent │
         └────────┬────────┘
                  │
          ┌───────▼────────┐
          │  out_of_scope? │  (conditional edge)
          └──┬─────────┬───┘
    medical  │         │  out_of_scope
             ▼         ▼
    retrieve_context  refuse_response
             │
             ▼
    generate_response
             │
             ▼
    validate_response ──→ END
```

The 4 mandatory nodes from the milestone (`classify_intent → retrieve_context → generate_response → validate_response`) are all present. `refuse_response` is the 5th node reached only on the out-of-scope branch.

### Shared State

```python
class AssistantState(TypedDict):
    query: str                    # user input
    intent: str                   # "medical" | "out_of_scope"
    retrieved_docs: list[Document]
    ml_context: str | None        # ML prediction injected via --features
    response: str
    sources: list[str]
    refused: bool
    error: str | None             # set on classify_intent failure; defaults routing to out_of_scope
```

---

## LLM Integration Strategy

**M3 (now):** Use `ChatGoogleGenerativeAI` from `langchain-google-genai` — this is the LangChain-native Gemini wrapper, NOT the existing `GeminiClient` from `src/llm/client.py`.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def get_chat_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )
```

> `src/llm/client.py` (`GeminiClient`, `MockClient`) is used by the existing Phase 1/2 explanation scripts only — it is NOT LangChain-compatible and must NOT be passed to LangChain chains or LangGraph nodes.

**After M2:** Swap the `generate_response` node to use `HuggingFacePipeline` (from `langchain-huggingface`) wrapping the PeftModel + TinyLlama. `classify_intent` and `validate_response` always use `ChatGoogleGenerativeAI`.

---

## Components

### `src/assistant/retriever.py`

Two public functions:

```python
def build_vectorstore(kb_dir: Path, persist_dir: Path) -> Chroma:
    """Load .txt files from kb_dir, chunk, embed, persist to persist_dir."""

def get_retriever(persist_dir: Path, k: int = 3) -> VectorStoreRetriever:
    """Load existing vectorstore, return top-k retriever."""
```

Embeddings: `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")` — local, no API key.

### `src/assistant/chain.py`

LCEL chain with explicit input schema:

```python
def build_chain(retriever, llm) -> Runnable:
    """
    Returns a Runnable that accepts:
        {"question": str, "ml_context": str | None}
    and returns a str (the generated answer).

    Internally: retriever runs on "question", ml_context is passed through.
    """
```

Prompt template fields: `{context}` (retrieved docs), `{question}`, `{ml_context}` (ML prediction string or empty string if None).

LCEL wiring:
```python
chain = (
    RunnableParallel(
        context=(itemgetter("question") | retriever | format_docs),
        question=itemgetter("question"),
        ml_context=RunnableLambda(lambda x: x.get("ml_context") or ""),  # None → ""
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

`None → ""` coercion happens inside `build_chain()` via `RunnableLambda`, so callers never need to handle it.

### `src/assistant/graph.py`

LangGraph `StateGraph` with 5 nodes:

| Node | Responsibility |
|---|---|
| `classify_intent` | `ChatGoogleGenerativeAI` prompt → sets `intent` to `"medical"` or `"out_of_scope"`. On any exception or unrecognized output: set `error`, default `intent` to `"out_of_scope"` (fail-safe). |
| `retrieve_context` | ChromaDB top-3 docs + prepend `ml_context` string if present |
| `generate_response` | `build_chain()` with `ChatGoogleGenerativeAI` |
| `validate_response` | Append disclaimer + extract source filenames into `sources` |
| `refuse_response` | Set `response` to standard refusal, `refused=True` |

Conditional edge after `classify_intent`:
```python
def route_intent(state: AssistantState) -> str:
    if state["intent"] == "medical":
        return "retrieve_context"
    return "refuse_response"
```

Public function: `build_graph() -> CompiledGraph`

Logging: each `build_graph()` invocation creates a logger via `setup_json_logger("assistant", ASSISTANT_LOG_PATH)`. After `validate_response`, call `log_event(logger, "interaction", query=..., intent=..., sources=..., response_len=..., refused=...)`.

### `scripts/build_kb.py` (new script)

One-time script to populate `data/kb/`:
1. Parse MedQuAD XML files from `MEDQUAD_DIR` (defined in `src/config.py`)
2. Filter for oncology/cancer topics
3. Write one `.txt` per Q&A pair to `KB_DIR` (defined in `src/config.py`)
4. Call `build_vectorstore(KB_DIR, VECTORSTORE_DIR)` to index into ChromaDB

### `scripts/run_assistant.py`

Interactive CLI:

```
python scripts/run_assistant.py
python scripts/run_assistant.py --features "17.99,10.38,122.8,..."
```

- `--features`: 30 comma-separated Wisconsin feature values
  → load `artifacts/models/best_model_with_threshold.joblib` (path: `BASE_DIR / "artifacts" / "models" / "best_model_with_threshold.joblib"` — add `BEST_MODEL_PATH` constant to `src/config.py`)
  → run prediction → format as `"ML Pipeline prediction: {prob:.0%} probability of malignancy ({label} — RF model)"`
  → store in `AssistantState.ml_context`
- Interactive loop: reads query from stdin → `graph.invoke(state)` → prints response + sources
- `Ctrl+C` to exit

---

## Knowledge Base

**Source:** MedQuAD — `1_CancerGov_QA/` subfolder (cancer.gov Q&A)
**Path:** `MEDQUAD_DIR` from `src/config.py` = `data/external/medquad/1_CancerGov_QA/`

**Population flow:**
```
MEDQUAD_DIR (XML) → scripts/build_kb.py → KB_DIR / data/kb/ (.txt) → VECTORSTORE_DIR / artifacts/vectorstore/ (ChromaDB)
```

---

## ML Integration

When `--features` is provided:
1. Parse 30 floats from CLI argument
2. Load model from `BEST_MODEL_PATH` (RF from Phase 1/2)
3. Run prediction → probability + label
4. Format: `"ML Pipeline prediction: 87% probability of malignancy (Malignant — RF model)"`
5. Store in `AssistantState.ml_context`
6. `retrieve_context` node prepends this string to the retrieved docs context

---

## Logging

Use `src/logging_utils.py` (`setup_json_logger`, `log_event`) — same pattern as existing pipeline.

| Log file | Path constant | Schema |
|---|---|---|
| `assistant.jsonl` | `ASSISTANT_LOG_PATH` | `stage, query, intent, sources, response_len, refused` |

- **Happy path:** logged inside `validate_response` (after sources are extracted).
- **Refused path:** also logged, inside `refuse_response` (with `sources=[]`, `refused=True`). Refused interactions must be logged — they reveal out-of-scope query patterns.

---

## Data Flow (full interaction)

```
User query + optional --features
  → [classify_intent]    ChatGoogleGenerativeAI → "medical" or "out_of_scope" (fail-safe: out_of_scope)
  → [retrieve_context]   ChromaDB top-3 + ml_context prepended
  → [generate_response]  ChatGoogleGenerativeAI generates grounded answer
  → [validate_response]  Disclaimer appended + sources extracted + ASSISTANT_LOG_PATH written
  → Output: answer + sources + disclaimer
```

Out-of-scope path:
```
User query (e.g. "what is the weather today?")
  → [classify_intent]    → "out_of_scope"
  → [refuse_response]    → "Posso responder apenas perguntas médicas relacionadas a oncologia."
  → Output: refusal message
```

---

## Files to Create/Modify

| File | Status | Action |
|---|---|---|
| `src/assistant/retriever.py` | stub | implement `build_vectorstore()` + `get_retriever()` |
| `src/assistant/chain.py` | stub | implement `build_chain()` with dict input schema |
| `src/assistant/graph.py` | stub | implement `build_graph()`: 5 nodes, conditional routing, logging |
| `scripts/build_kb.py` | missing | create: MedQuAD XML parser + vectorstore builder |
| `scripts/run_assistant.py` | raises | implement interactive CLI with `--features` |
| `src/config.py` | exists | add `BEST_MODEL_PATH` constant |
| `requirements.txt` | exists | already updated: added `sentence-transformers`, `langchain-huggingface` |

**Out of scope for M3** (M4):
- `src/assistant/guardrails.py`
- `src/assistant/audit_logger.py`
- `src/assistant/explainer.py`
- `src/assistant/tools.py` (deferred to future milestone)

---

## Dependencies

```
# Already in requirements.txt (updated):
langchain>=0.2.0,<0.4.0
langchain-community>=0.2.0,<0.4.0
langchain-google-genai>=1.0.0,<2.0.0
langchain-huggingface>=0.0.3,<0.2.0
langgraph>=0.1.0,<0.3.0
chromadb>=0.5.0,<0.6.0
sentence-transformers>=2.7.0,<4.0.0
```

---

## Success Criteria

- `python scripts/run_assistant.py` responde perguntas médicas com contexto recuperado do KB
- LangGraph executa os 5 nós: `classify_intent → retrieve_context → generate_response → validate_response` (happy path) + `refuse_response` (out-of-scope path)
- Resposta inclui citação da fonte recuperada (`sources` field)
- Query fora do escopo retorna recusa (routing condicional funcionando)
- `--features` injeta predição ML no contexto da resposta
- `artifacts/logs/assistant.jsonl` cresce a cada interação
