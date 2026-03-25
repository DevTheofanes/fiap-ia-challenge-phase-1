# Fix Deprecation Warnings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate two warnings that appear on every run of `scripts/run_assistant.py` by migrating the Chroma import to `langchain-chroma` and silencing the harmless `embeddings.position_ids` log from `transformers`.

**Architecture:** Two surgical changes in one file (`src/assistant/retriever.py`) plus one line in `requirements.txt`. No logic, interface, or behavior changes.

**Tech Stack:** Python stdlib `logging`, `langchain-chroma`, `langchain-community` (unchanged for other imports), `chromadb` (unchanged).

**Spec:** `docs/superpowers/specs/2026-03-24-fix-deprecation-warnings-design.md`

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `requirements.txt` | Add `langchain-chroma>=0.2.0,<2.0.0` |
| Modify | `src/assistant/retriever.py` | Add `import logging`; swap Chroma import; add `setLevel` in `_get_embeddings()` |

> **Execution order matters:** Complete Task 1 (`requirements.txt`) before Task 2 (`retriever.py`) so that `langchain_chroma` is installed before the import verification step.

---

### Task 1: Add `langchain-chroma` to requirements and install

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Open `requirements.txt` and locate the Phase 3 section**

  It starts with:
  ```
  # ── Phase 3: LangChain + Vector Store ──────────────────────────
  ```

- [ ] **Step 2: Add `langchain-chroma` to the Phase 3 section**

  Insert after the `langchain-text-splitters` line. The full Phase 3 block after the edit:
  ```
  # ── Phase 3: LangChain + Vector Store ──────────────────────────
  langchain>=1.2.0,<3.0.0
  langchain-community>=0.4.0,<2.0.0
  langchain-google-genai>=4.2.0,<6.0.0
  langchain-huggingface>=1.2.0,<3.0.0
  langchain-text-splitters>=0.3.0,<1.0.0
  langchain-chroma>=0.2.0,<2.0.0  # >=0.2.0 required for Python 3.14 compat
  langgraph>=1.1.0,<3.0.0
  chromadb>=1.5.0,<3.0.0
  sentence-transformers>=2.7.0,<4.0.0
  ```

- [ ] **Step 3: Install the new dependency**

  ```bash
  pip install -r requirements.txt
  ```
  Expected: installs `langchain-chroma` (and any missing deps) without errors.

- [ ] **Step 4: Commit**

  ```bash
  git add requirements.txt
  git commit -m "chore: add langchain-chroma dependency"
  ```
  (Kept as a separate commit from the code change so the dependency update is independently reviewable.)

---

### Task 2: Migrate Chroma import and suppress position_ids log

**Files:**
- Modify: `src/assistant/retriever.py`

**Background:**
- `from langchain_community.vectorstores import Chroma` is deprecated since LangChain 0.2.9. Replace with `from langchain_chroma import Chroma`.
- `HuggingFaceEmbeddings` triggers a `transformers.modeling_utils` log line (`embeddings.position_ids UNEXPECTED`) via Python's `logging` module (not `warnings.warn`). Silence it with `logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)` inside `_get_embeddings()`, which is the only function that constructs `HuggingFaceEmbeddings` — called from both `build_vectorstore` and `get_retriever`.

- [ ] **Step 1: Open `src/assistant/retriever.py` and read its current contents**

  Verify the file starts like this (lines 6–13 approximately):
  ```python
  from __future__ import annotations

  from pathlib import Path

  from langchain_community.document_loaders import DirectoryLoader, TextLoader
  from langchain_community.vectorstores import Chroma          # ← to replace
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  ```

- [ ] **Step 2: Apply changes to `src/assistant/retriever.py`**

  Make three edits:

  **2a.** Add `import logging` alongside the existing stdlib import (`from pathlib import Path`):
  ```python
  # Before
  from pathlib import Path

  # After
  import logging
  from pathlib import Path
  ```

  **2b.** Replace the deprecated Chroma import:
  ```python
  # Before
  from langchain_community.vectorstores import Chroma
  # After
  from langchain_chroma import Chroma
  ```

  **2c.** Update `_get_embeddings()` to suppress the log before constructing the model:
  ```python
  # Before
  def _get_embeddings() -> HuggingFaceEmbeddings:
      return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

  # After
  def _get_embeddings() -> HuggingFaceEmbeddings:
      # Suppress harmless checkpoint key mismatch log from transformers/BertModel
      logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
      return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
  ```

- [ ] **Step 3: Verify the module imports cleanly**

  Run:
  ```bash
  python -c "from src.assistant.retriever import get_retriever, build_vectorstore; print('OK')"
  ```
  Expected output:
  ```
  OK
  ```

- [ ] **Step 4: Commit**

  ```bash
  git add src/assistant/retriever.py
  git commit -m "fix: migrate Chroma import to langchain-chroma, silence position_ids log"
  ```

---

### Task 3: End-to-end verification

**Files:** none modified

- [ ] **Step 1: Rebuild the vectorstore**

  ```bash
  python scripts/build_kb.py
  ```
  This ensures `_get_embeddings()` is actually called (model loaded), which is the trigger for the `embeddings.position_ids` warning.

- [ ] **Step 2: Run the assistant and inspect output**

  ```bash
  python scripts/run_assistant.py 2>&1 | head -30
  ```

- [ ] **Step 3: Confirm both warnings are gone**

  The following strings must NOT appear in the output:
  - `embeddings.position_ids`
  - `LangChainDeprecationWarning`
  - `Chroma was deprecated`

  If `embeddings.position_ids` still appears, verify that `_get_embeddings()` contains the `setLevel` call and that `import logging` is at the top of `retriever.py`.

  If `Chroma was deprecated` still appears, verify the import in `retriever.py` reads `from langchain_chroma import Chroma` (not `langchain_community`).
