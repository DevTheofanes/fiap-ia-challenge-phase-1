# Fix Deprecation Warnings — Design Spec

**Date:** 2026-03-24
**Scope:** `src/assistant/retriever.py`, `requirements.txt`

## Problem

Two warnings appear on every run of `scripts/run_assistant.py`:

1. **Warning #3 — `embeddings.position_ids UNEXPECTED`**: When loading `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`, the `transformers` library emits a log line (via Python's `logging` module — not `warnings.warn`) flagging an unexpected checkpoint key. This is a known, harmless mismatch between sentence-transformers checkpoints and the bare `BertModel` architecture. No functional impact.

2. **Warning #4 — `Chroma` deprecated in `langchain_community`**: `langchain_community.vectorstores.Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. The replacement is the `langchain-chroma` package.

## Approach

**Option A (chosen):** Silence warning #3 via the `logging` module inside `_get_embeddings()` + migrate Chroma import to `langchain-chroma`.

Rationale: surgical changes with zero logic impact. The logging suppression is scoped to `transformers.modeling_utils` and placed inside `_get_embeddings()` — the only function that constructs `HuggingFaceEmbeddings`. Note: `logging.setLevel` mutates the global logger for the process lifetime; this is acceptable because the project runs as short-lived scripts and `transformers.modeling_utils` WARNING-level output has no diagnostic value in this context. `_get_embeddings()` is called from both `build_vectorstore` and `get_retriever`, so the suppression fires on both paths — this is intentional and correct.

## Changes

### `src/assistant/retriever.py`

1. Replace import:
   ```python
   # Before
   from langchain_community.vectorstores import Chroma
   # After
   from langchain_chroma import Chroma
   ```

2. Add `import logging` at the top of the file (with the other stdlib imports). Then, inside `_get_embeddings()`, call `setLevel` before constructing `HuggingFaceEmbeddings`:
   ```python
   # At the top of the file (with other imports):
   import logging

   # Inside _get_embeddings():
   def _get_embeddings() -> HuggingFaceEmbeddings:
       # Suppress harmless checkpoint key mismatch log from transformers/BertModel
       logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
       return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
   ```

### `requirements.txt`

Add to the Phase 3 section:
```
langchain-chroma>=0.1.0,<2.0.0
```

The existing `chromadb` entry remains — `langchain-chroma` wraps it. Upper bound is `<2.0.0`, consistent with the project's pinning convention for Phase 3 dependencies.

## Out of Scope

- Warning #1 (Pydantic V1 + Python 3.14): requires upstream LangChain fix.
- Warning #2 (HF Hub unauthenticated): resolved by setting `HF_TOKEN` env var — user's choice.
- No logic, interface, or behavior changes.

## Testing

1. Run `pip install -r requirements.txt` to install `langchain-chroma`.
2. Run `python scripts/build_kb.py` to rebuild the vectorstore (ensures the embedding model is actually loaded).
3. Run `python scripts/run_assistant.py`.
4. Confirm the following lines are absent from the output:
   - Any line containing `embeddings.position_ids`
   - Any line containing `LangChainDeprecationWarning` or `Chroma was deprecated`

Note: there is no automated regression test for these suppressions, consistent with the project's testing philosophy (validation via pipeline scripts, per CLAUDE.md).
