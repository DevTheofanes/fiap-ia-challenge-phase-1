# M3 Medical Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-powered medical assistant with RAG over MedQuAD, conditional routing for out-of-scope queries, ML prediction injection, and interaction logging.

**Architecture:** 5-node LangGraph StateGraph (classify_intent → retrieve_context → generate_response → validate_response, with a refuse_response branch for out-of-scope queries). ChromaDB stores MedQuAD oncology docs as the knowledge base. ChatGoogleGenerativeAI (langchain-google-genai) is used for all LLM calls — NOT the existing GeminiClient from src/llm/client.py.

**Tech Stack:** `langgraph`, `langchain`, `langchain-google-genai`, `chromadb`, `sentence-transformers`, `joblib` (ML model), Python 3.11+

---

## Prerequisites — Install dependencies first

Before starting any task, install all required packages:

```bash
pip install langchain langchain-community langchain-google-genai langchain-huggingface langgraph chromadb sentence-transformers
```

---

## File Map

| File | Status | Role |
|---|---|---|
| `src/config.py` | modify | add `BEST_MODEL_PATH` constant |
| `src/assistant/retriever.py` | stub → implement | ChromaDB vectorstore builder + retriever factory |
| `src/assistant/chain.py` | stub → implement | LCEL RAG chain with ml_context support |
| `src/assistant/graph.py` | stub → implement | LangGraph StateGraph: 5 nodes, routing, logging |
| `scripts/build_kb.py` | create | one-time: clone MedQuAD, parse XML, write KB, build vectorstore |
| `scripts/run_assistant.py` | stub → implement | interactive CLI with optional `--features` for ML context |

**Not touched in M3:** `guardrails.py`, `audit_logger.py`, `explainer.py`, `tools.py`

---

## Task 1: Add BEST_MODEL_PATH to config.py

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add the constant**

Open `src/config.py` and append after the last existing line (after `MEDQUAD_DIR`):

```python
BEST_MODEL_PATH = ARTIFACTS_DIR / "models" / "best_model_with_threshold.joblib"
```

- [ ] **Step 2: Verify the path resolves correctly**

```bash
python -c "from src.config import BEST_MODEL_PATH; print(BEST_MODEL_PATH); print(BEST_MODEL_PATH.exists())"
```

Expected output:
```
/path/to/fiap-ia-challenge-phase-2/artifacts/models/best_model_with_threshold.joblib
True
```

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat(config): add BEST_MODEL_PATH constant for M3 ML integration"
```

---

## Task 2: Implement retriever.py

**Files:**
- Modify: `src/assistant/retriever.py`

- [ ] **Step 1: Write the implementation**

Replace the contents of `src/assistant/retriever.py` with:

```python
"""Vector store retriever for the medical knowledge base (M3).

Indexes documents from data/kb/ using ChromaDB and retrieves
relevant context for user queries.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)


def build_vectorstore(kb_dir: Path, persist_dir: Path) -> Chroma:
    """Load .txt files from kb_dir, chunk, embed, and persist to persist_dir.

    Args:
        kb_dir: Directory containing .txt knowledge base files.
        persist_dir: Directory where ChromaDB will persist the index.

    Returns:
        Loaded Chroma vectorstore.
    """
    loader = DirectoryLoader(str(kb_dir), glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=str(persist_dir),
    )
    return vectorstore


def get_retriever(persist_dir: Path, k: int = 3):
    """Load an existing ChromaDB vectorstore and return a top-k retriever.

    Args:
        persist_dir: Directory where the ChromaDB index is persisted.
        k: Number of documents to retrieve per query.

    Returns:
        VectorStoreRetriever configured for top-k search.
    """
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=_get_embeddings(),
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from src.assistant.retriever import build_vectorstore, get_retriever; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/assistant/retriever.py
git commit -m "feat(M3): implement retriever — ChromaDB vectorstore builder and retriever factory"
```

---

## Task 3: Implement chain.py

**Files:**
- Modify: `src/assistant/chain.py`

- [ ] **Step 1: Write the implementation**

Replace the contents of `src/assistant/chain.py` with:

```python
"""LangChain LCEL chain for the medical assistant (M3).

Combines the retriever with ChatGoogleGenerativeAI to generate
grounded responses from retrieved context.
"""
from __future__ import annotations

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a medical assistant helping doctors with clinical questions about oncology.
Use ONLY the retrieved knowledge below to answer. If the context does not contain \
enough information to answer, say so clearly. Never prescribe medications or make \
definitive diagnoses — always recommend consulting a qualified physician.

{ml_context_block}
Retrieved Knowledge:
{context}

Question: {question}

Answer in the same language as the question. Be concise and cite relevant details \
from the retrieved context."""
)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _format_ml_context(ml_context: str) -> str:
    if not ml_context:
        return ""
    return f"ML Pipeline Context:\n{ml_context}\n"


def build_chain(retriever, llm):
    """Build a RAG chain that accepts a dict input.

    Input schema: {"question": str, "ml_context": str | None}
    Output: str (the generated answer)

    None → "" coercion for ml_context happens inside this function.
    """
    chain = (
        RunnableParallel(
            context=(itemgetter("question") | retriever | _format_docs),
            question=itemgetter("question"),
            ml_context_block=RunnableLambda(
                lambda x: _format_ml_context(x.get("ml_context") or "")
            ),
        )
        | _PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python -c "from src.assistant.chain import build_chain; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/assistant/chain.py
git commit -m "feat(M3): implement LCEL RAG chain with ml_context support"
```

---

## Task 4: Implement graph.py

**Files:**
- Modify: `src/assistant/graph.py`

- [ ] **Step 1: Write the implementation**

Replace the contents of `src/assistant/graph.py` with:

```python
"""LangGraph StateGraph for the medical assistant workflow (M3).

Nodes: classify_intent → retrieve_context → generate_response → validate_response
       classify_intent → refuse_response  (out-of-scope branch)
"""
from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.assistant.chain import build_chain
from src.assistant.retriever import get_retriever
from src.config import ASSISTANT_LOG_PATH, VECTORSTORE_DIR
from src.logging_utils import log_event, setup_json_logger

_DISCLAIMER = (
    "\n\n⚠ Esta informação é educacional e não substitui consulta médica profissional."
)
_REFUSAL_MSG = (
    "Posso responder apenas perguntas médicas relacionadas a oncologia e diagnóstico. "
    "Por favor, reformule sua pergunta dentro desse escopo."
)
_CLASSIFY_PROMPT = (
    "Classify the following question as either 'medical' (related to medicine, "
    "oncology, cancer, diagnosis, treatment, symptoms, anatomy, or clinical topics) "
    "or 'out_of_scope' (anything else).\n"
    "Reply with ONLY the single word: medical OR out_of_scope\n\n"
    "Question: {query}"
)


class AssistantState(TypedDict):
    query: str
    intent: str                    # "medical" | "out_of_scope"
    retrieved_docs: list[Document]
    ml_context: str | None
    response: str
    sources: list[str]
    refused: bool
    error: str | None


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )


def build_graph():
    """Build and compile the LangGraph medical assistant StateGraph.

    Returns:
        CompiledGraph ready for .invoke()
    """
    logger = setup_json_logger("assistant", ASSISTANT_LOG_PATH)
    retriever = get_retriever(VECTORSTORE_DIR)
    llm = _get_llm()
    chain = build_chain(retriever, llm)

    # ── Node functions ────────────────────────────────────────────

    def classify_intent(state: AssistantState) -> AssistantState:
        try:
            prompt = _CLASSIFY_PROMPT.format(query=state["query"])
            result = llm.invoke(prompt)
            text = result.content.strip().lower()
            intent = "medical" if "medical" in text else "out_of_scope"
        except Exception as exc:
            intent = "out_of_scope"
            return {**state, "intent": intent, "error": str(exc)}
        return {**state, "intent": intent, "error": None}

    def retrieve_context(state: AssistantState) -> AssistantState:
        docs = retriever.invoke(state["query"])
        return {**state, "retrieved_docs": docs}

    def generate_response(state: AssistantState) -> AssistantState:
        answer = chain.invoke(
            {"question": state["query"], "ml_context": state["ml_context"]}
        )
        return {**state, "response": answer}

    def validate_response(state: AssistantState) -> AssistantState:
        sources = [
            doc.metadata.get("source", "unknown")
            for doc in state.get("retrieved_docs", [])
        ]
        response = state["response"] + _DISCLAIMER
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

    def refuse_response(state: AssistantState) -> AssistantState:
        log_event(
            logger,
            "interaction",
            query=state["query"],
            intent=state["intent"],
            sources=[],
            response_len=len(_REFUSAL_MSG),
            refused=True,
        )
        return {**state, "response": _REFUSAL_MSG, "sources": [], "refused": True}

    def route_intent(state: AssistantState) -> str:
        if state.get("intent") == "medical":
            return "retrieve_context"
        return "refuse_response"

    # ── Build graph ───────────────────────────────────────────────

    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("validate_response", validate_response)
    graph.add_node("refuse_response", refuse_response)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "validate_response")
    graph.add_edge("validate_response", END)
    graph.add_edge("refuse_response", END)

    return graph.compile()
```

- [ ] **Step 2: Verify it imports cleanly (vectorstore may not exist yet — that's expected)**

```bash
python -c "from src.assistant.graph import build_graph, AssistantState; print('OK')"
```

Expected: `OK` (import only — no invocation yet)

- [ ] **Step 3: Commit**

```bash
git add src/assistant/graph.py
git commit -m "feat(M3): implement LangGraph StateGraph — 5 nodes, conditional routing, logging"
```

---

## Task 5: Create scripts/build_kb.py

**Files:**
- Create: `scripts/build_kb.py`

- [ ] **Step 1: Write the script**

Create `scripts/build_kb.py`:

```python
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
from pathlib import Path

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
    if _MEDQUAD_ROOT.exists():
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
    """Parse MedQuAD cancer Q&A and write to KB_DIR. Returns count of files written."""
    if not MEDQUAD_DIR.exists():
        print(f"ERROR: MedQuAD source not found at {MEDQUAD_DIR}", file=sys.stderr)
        print("Run with a network connection so the repo can be cloned.", file=sys.stderr)
        sys.exit(1)

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
```

- [ ] **Step 2: Run the script**

```bash
python scripts/build_kb.py
```

Expected output (approximately):
```
Cloning MedQuAD to .../data/external/medquad …
Clone complete.
Found N XML files in .../data/external/medquad/1_CancerGov_QA
Written N Q&A files to .../data/kb
Building ChromaDB vectorstore at .../artifacts/vectorstore …
Vectorstore built successfully.
```

> **Layout check:** After cloning, verify `data/external/medquad/1_CancerGov_QA/` exists and contains `.xml` files: `ls data/external/medquad/1_CancerGov_QA/`. If the path differs, update `MEDQUAD_DIR` in `src/config.py` accordingly.

- [ ] **Step 4: Verify KB and vectorstore were created**

```bash
python -c "
from src.config import KB_DIR, VECTORSTORE_DIR
txt_files = list(KB_DIR.glob('*.txt'))
print(f'KB files: {len(txt_files)}')
print(f'Vectorstore exists: {VECTORSTORE_DIR.exists()}')
assert len(txt_files) > 0, 'No KB files found!'
assert VECTORSTORE_DIR.exists(), 'Vectorstore not created!'
print('OK')
"
```

Expected: KB files > 0, vectorstore exists.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_kb.py
git commit -m "feat(M3): add build_kb.py — MedQuAD parser and ChromaDB vectorstore builder"
```

---

## Task 6: Implement scripts/run_assistant.py

**Files:**
- Modify: `scripts/run_assistant.py`

- [ ] **Step 1: Write the implementation**

Replace the contents of `scripts/run_assistant.py` with:

```python
#!/usr/bin/env python
"""Interactive medical assistant CLI (Phase 3 — M3).

Usage:
    python scripts/run_assistant.py
    python scripts/run_assistant.py --features "17.99,10.38,122.8,1001,0.118,..."

The --features flag accepts 30 comma-separated float values (Wisconsin dataset
feature order) and injects an ML prediction into the assistant context.
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np

from src.assistant.graph import build_graph
from src.config import BEST_MODEL_PATH


def _parse_features(features_str: str) -> str | None:
    """Parse --features string, run ML model, return formatted prediction string."""
    try:
        values = [float(v.strip()) for v in features_str.split(",")]
    except ValueError as e:
        print(f"ERROR: Could not parse --features: {e}", file=sys.stderr)
        return None

    if len(values) != 30:
        print(f"ERROR: Expected 30 features, got {len(values)}", file=sys.stderr)
        return None

    if not BEST_MODEL_PATH.exists():
        print(f"WARNING: ML model not found at {BEST_MODEL_PATH}. Skipping ML context.")
        return None

    model = joblib.load(BEST_MODEL_PATH)
    X = np.array(values).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]
    label = "Malignant" if prob >= 0.5 else "Benign"
    return f"ML Pipeline prediction: {prob:.0%} probability of malignancy ({label} — RF model)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical assistant CLI (M3)")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="30 comma-separated Wisconsin feature values for ML prediction context",
    )
    args = parser.parse_args()

    ml_context: str | None = None
    if args.features:
        ml_context = _parse_features(args.features)
        if ml_context:
            print(f"ML context: {ml_context}\n")

    print("Building assistant graph …")
    graph = build_graph()
    print("Assistant ready. Type your question (Ctrl+C to exit).\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue

        state = {
            "query": query,
            "intent": "",
            "retrieved_docs": [],
            "ml_context": ml_context,
            "response": "",
            "sources": [],
            "refused": False,
            "error": None,
        }

        result = graph.invoke(state)

        print(f"\nAssistant: {result['response']}")

        if result.get("sources"):
            unique_sources = sorted(set(result["sources"]))
            print(f"Sources: {', '.join(unique_sources)}")

        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — verify it starts without error**

```bash
python scripts/run_assistant.py --help
```

Expected: prints usage/help with `--features` flag.

- [ ] **Step 3: End-to-end test — ask a medical question**

```bash
echo "What are the risk factors for breast cancer?" | python scripts/run_assistant.py
```

Expected: assistant responds with an answer + source filenames + disclaimer. No Python traceback.

- [ ] **Step 4: Test out-of-scope routing**

```bash
echo "What is the capital of France?" | python scripts/run_assistant.py
```

Expected: refusal message (`Posso responder apenas perguntas médicas…`)

- [ ] **Step 5: Test ML context injection**

```bash
echo "What treatment should I consider given these findings?" | python scripts/run_assistant.py --features "17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189"
```

Expected: response begins with ML context printed, answer references the prediction probability.

- [ ] **Step 6: Verify assistant.jsonl grows after interactions**

```bash
python -c "
from src.config import ASSISTANT_LOG_PATH
import json
lines = ASSISTANT_LOG_PATH.read_text().strip().splitlines()
print(f'Log entries: {len(lines)}')
print(json.loads(lines[-1]))
"
```

Expected: at least 1 log entry with fields: `stage`, `query`, `intent`, `sources`, `response_len`, `refused`.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_assistant.py
git commit -m "feat(M3): implement run_assistant.py — interactive CLI with ML context injection"
```

---

## Task 7: Final integration check

- [ ] **Step 1: Run full happy-path interaction**

```bash
python scripts/run_assistant.py
```

Type: `What are the symptoms of invasive ductal carcinoma?`

Verify:
- Response contains medical content
- Sources line lists `.txt` filenames from `data/kb/`
- Disclaimer `⚠ Esta informação é educacional…` appears

- [ ] **Step 2: Verify all 5 nodes executed (check log)**

```bash
python -c "
from src.config import ASSISTANT_LOG_PATH
import json
for line in ASSISTANT_LOG_PATH.read_text().strip().splitlines():
    entry = json.loads(line)
    print(entry.get('intent'), '|', entry.get('refused'), '|', entry.get('response_len'))
"
```

Expected: entries with `intent=medical, refused=False` (happy path) and `intent=out_of_scope, refused=True` (refusal path, from Task 6 Step 4).

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat(M3): complete medical assistant pipeline — LangGraph + RAG + MedQuAD KB"
```

---

## Troubleshooting

**`ChromaDB` import error:** `pip install chromadb`

**`sentence-transformers` not found:** `pip install sentence-transformers`

**`GEMINI_API_KEY` not set:** copy `.env.example` to `.env` and fill in the key. Verify with:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GEMINI_API_KEY', 'NOT SET')[:8])"
```

**`build_graph()` fails with `vectorstore not found`:** run `python scripts/build_kb.py` first.

**MedQuAD clone fails (no network):** manually place XML files in `data/external/medquad/1_CancerGov_QA/` and re-run `python scripts/build_kb.py`.

**MedQuAD layout mismatch after clone:** run `ls data/external/medquad/` to see the actual structure, then update `MEDQUAD_DIR` in `src/config.py` to point to the correct subfolder containing `.xml` files.

**After M2 completes — swapping to TinyLlama:** in `graph.py`, replace `_get_llm()` in `generate_response` with:
```python
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline as hf_pipeline
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import FINETUNE_FINAL_ADAPTER_DIR

config = PeftConfig.from_pretrained(str(FINETUNE_FINAL_ADAPTER_DIR))
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, str(FINETUNE_FINAL_ADAPTER_DIR))
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
generate_llm = HuggingFacePipeline(pipeline=pipe)
```
