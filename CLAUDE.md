# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML pipeline for breast cancer diagnosis (Wisconsin dataset) with three stages:
1. **Baseline training** — standard scikit-learn classifiers (LR, RF, SVM, KNN)
2. **Genetic Algorithm optimization** — custom GA for hyperparameter search
3. **LLM explanations** — Gemini-based clinical interpretation of predictions

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Set `LLM_USE_MOCK=true` to skip real API calls during development.

## Common Commands

```bash
# Full baseline training and evaluation
python scripts/run_baseline.py

# Run GA optimization (model: LR|RF|SVM|KNN, exp: A|B|C)
python scripts/run_ga_experiments.py --model LR --exp A --seed 42
python scripts/run_ga.py --model RF --exp B --seed 42

# Generate LLM explanations
python scripts/explain_sample.py --split val --sample-idx 0

# Aggregate results
python scripts/summarize_ga_runs.py
python scripts/summarize_results.py

# Generate phase 2 technical report
python scripts/generate_phase2_report.py

# Run fine-tuning (M2)
python scripts/fine_tune.py

# Evaluate fine-tuned model (M2)
python scripts/eval_finetune.py

# Run medical assistant (M3, after implementation)
python scripts/run_assistant.py

# Docker
docker build -t fiap-ia-challenge .
docker run --rm -v "$(pwd)/data:/app/data" fiap-ia-challenge
```

There is no test suite. Validation is done by running the pipeline scripts and inspecting `artifacts/`.

## Architecture

### Data Flow

```
data/wisconsin_breast_cancer.csv
  → src/data/load.py + preprocess.py   (drop ID cols, median impute, M/B→1/0, StandardScaler)
  → src/data/split.py                  (60/20/20 stratified, cached to data/splits.npz)
  → scripts/run_baseline.py            (train defaults, save to artifacts/baseline_models/)
  → scripts/run_ga_experiments.py      (GA optimization, results in artifacts/ga_runs/)
  → scripts/summarize_ga_runs.py       (aggregated metrics in artifacts/ga_summary/)
  → scripts/explain_sample.py          (LLM explanations in artifacts/llm/)
  → scripts/fine_tune.py               (LoRA fine-tuning on data/finetune/, checkpoints in artifacts/finetune_checkpoints/)
  → scripts/run_assistant.py           (LangGraph assistant, vectorstore in artifacts/vectorstore/)
```

### Key Modules

| Path | Role |
|------|------|
| `src/config.py` | All paths, constants, splits ratios, random seed (42) |
| `src/logging_utils.py` | Structured JSONL logging to `artifacts/logs/` |
| `src/utils.py` | Shared utilities: `set_seeds()` |
| `src/data/` | Load, preprocess, split |
| `src/models/` | Model registry, training, evaluation wrapper |
| `src/genetic/` | GA operators: `encoding.py`, `ga.py`, `fitness.py`, `search_space.py` |
| `src/evaluation/` | Metrics (accuracy, F1, ROC-AUC, PR-AUC), stratified CV |
| `src/llm/` | Gemini client, prompt templates, structured JSON output with fallback |
| `src/assistant/retriever.py` | ChromaDB vector store for medical KB (M3) |
| `src/assistant/chain.py` | LangChain RetrievalQA chain (M3) |
| `src/assistant/graph.py` | LangGraph StateGraph workflow (M3) |
| `src/assistant/guardrails.py` | Input/output safety filters (M4) |
| `src/assistant/audit_logger.py` | Interaction audit trail (M4) |

### Genetic Algorithm

- Individuals are encoded as dicts of hyperparameters (see `src/genetic/encoding.py`)
- Fitness = F1-score via stratified cross-validation on training split (see `src/genetic/fitness.py`)
- Experiments A/B/C differ in GA configuration (population size, generations, mutation rate); defined in `docs/reports/Relatorio_Tecnico_Tech_Challenge_Fase2.md`
- Results written per run to `artifacts/ga_runs/<model>/<exp>/`

### Reproducibility

All experiments use `RANDOM_STATE = 42` from `src/config.py`. Stratified splits are cached to `data/splits.npz` on first run and reused thereafter. Pass `--seed` to GA scripts to override.

### LLM Integration

- Requires `GEMINI_API_KEY` and `GEMINI_MODEL` (default: `gemini-1.5-flash`)
- Set `LLM_USE_MOCK=true` to get deterministic mock responses without API calls
- Structured JSON output with graceful fallback if parsing fails
- Logs to `artifacts/logs/llm.jsonl`

### Documentation Structure

- `docs/reports/` — Technical reports (Fase 1, 2) and generated figures
- `docs/requirements/` — Phase specs, milestones, challenge PDFs
- `docs/ai/` — LLM evaluation rubrics, fine-tuning notes
- `docs/development/` — Architecture and developer guides
