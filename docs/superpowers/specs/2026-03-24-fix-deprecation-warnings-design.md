# Fix Deprecation Warnings — Design Spec

**Date:** 2026-03-24
**Scope:** `src/assistant/retriever.py`, `requirements.txt`

## Problem

Two warnings appear on every run of `scripts/run_assistant.py`:

1. **Warning #3 — `embeddings.position_ids UNEXPECTED`**: When loading `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`, the BertModel load report flags an unexpected key (`embeddings.position_ids`). This is a known, harmless mismatch between sentence-transformers checkpoints and the bare `BertModel` architecture. No functional impact.

2. **Warning #4 — `Chroma` deprecated in `langchain_community`**: `langchain_community.vectorstores.Chroma` was deprecated in LangChain 0.2.9 and will be removed in 1.0. The replacement is the `langchain-chroma` package.

## Approach

**Option A (chosen):** Suppress warning #3 locally in `retriever.py` + migrate Chroma import to `langchain-chroma`.

Rationale: surgical changes with zero logic impact. Warning filter is scoped to the specific message and module. Chroma migration resolves the actual deprecation.

## Changes

### `src/assistant/retriever.py`

1. Replace import:
   ```python
   # Before
   from langchain_community.vectorstores import Chroma
   # After
   from langchain_chroma import Chroma
   ```

2. Add warning filter after imports:
   ```python
   import warnings
   warnings.filterwarnings(
       "ignore",
       message=".*embeddings.position_ids.*",
       category=UserWarning,
   )
   ```

### `requirements.txt`

Add to the Phase 3 section:
```
langchain-chroma>=0.1.0,<1.0.0
```

The existing `chromadb` entry remains — `langchain-chroma` wraps it.

## Out of Scope

- Warning #1 (Pydantic V1 + Python 3.14): requires upstream LangChain fix.
- Warning #2 (HF Hub unauthenticated): resolved by setting `HF_TOKEN` env var — user's choice.
- No logic, interface, or behavior changes.

## Testing

Run `python scripts/run_assistant.py` and confirm neither warning appears in output.
