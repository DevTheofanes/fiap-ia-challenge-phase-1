# FIAP IA Tech Challenge - Phase 4 Multimodal Medical Assistant

This repository contains the full project evolution for the FIAP Tech Challenge, ending in a Phase 4 multimodal medical assistant that combines:

- Breast-cancer ML artifacts from earlier phases
- Fine-tuning of a TinyLlama-based local model
- LangGraph orchestration
- KB retrieval over medical documents
- Structured synthetic patient records
- Guardrails, citations, and audit logging
- Audio transcription and clinical transcript analysis
- YOLOv8 video screening for anomalous bleeding

## Phase 4 Architecture

```mermaid
flowchart LR
  A[classify_intent] -->|medical| B[retrieve_kb_context]
  B --> C[retrieve_patient_context]
  C --> D[process_audio]
  D --> E[process_video]
  E --> F[generate_response]
  F --> G[validate_response]
  A -->|out_of_scope| H[refuse_response]
```

Runtime policy:

- Primary assistant model: fine-tuned TinyLlama adapter
- Gemini: optional fallback only
- Default fallback behavior: disabled
- Audio transcription: OpenAI Whisper API
- Audio clinical analysis: GPT-4o-mini through `OpenAIClient`
- Video detection: YOLOv8 fine-tuned on synthetic `anomalous_bleeding` frames

## Project Structure

```text
.
├── artifacts/
│   ├── finetune_checkpoints/
│   ├── logs/
│   ├── yolo/
│   └── vectorstore/
├── data/
│   ├── finetune/
│   ├── kb/
│   ├── patients/
│   ├── synthetic_bleeding/
│   └── wisconsin_breast_cancer.csv
├── docs/
│   ├── reports/
│   └── requirements/
├── scripts/
│   ├── analyze_audio.py
│   ├── analyze_video.py
│   ├── build_kb.py
│   ├── eval_finetune.py
│   ├── fine_tune.py
│   ├── generate_synthetic_data.py
│   ├── prepare_finetune_data.py
│   ├── run_assistant.py
│   └── train_yolo.py
├── src/
│   ├── assistant/
│   ├── llm/
│   ├── models/
│   ├── multimodal/
│   └── genetic/
└── tests/
```

## Environment Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

3. Optional `.env` settings:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
ASSISTANT_ALLOW_FALLBACK=false
LLM_USE_MOCK=false
```

## Fine-Tuning Workflow

Prepare the instruction dataset:

```bash
python3 scripts/prepare_finetune_data.py --total 625 --seed 42
```

Train the LoRA adapter:

```bash
python3 scripts/fine_tune.py --epochs 1 --batch-size 2
```

Evaluate base vs fine-tuned model:

```bash
python3 scripts/eval_finetune.py --split val --num-samples 10
```

Artifacts:

- Adapter: `artifacts/finetune_checkpoints/final_adapter/`
- Logs: `artifacts/logs/finetune.jsonl`

## Phase 4 Audio Workflow

Analyze a consultation audio file:

```bash
python3 scripts/analyze_audio.py path/to/consultation.wav
```

Pipeline:

- `transcribe(audio_path)` calls the OpenAI transcription API with `whisper-1`.
- `analyze_transcript(transcript)` generates a structured clinical screening report with GPT-4o-mini.
- The report covers postpartum depression, anxiety, possible violence or unsafe conditions, and hormonal fatigue or sleep deprivation.

## Phase 4 Video Workflow

Generate the synthetic YOLOv8 dataset:

```bash
python3 scripts/generate_synthetic_data.py
```

Train YOLOv8:

```bash
python3 scripts/train_yolo.py --device mps
```

Use `--device cpu` if MPS or CUDA is unavailable.

Analyze a clinical video:

```bash
python3 scripts/analyze_video.py path/to/procedure.mp4 \
  --model-path artifacts/yolo/bleeding_yolov8n/weights/best.pt
```

Generated artifacts:

- Synthetic dataset: `data/synthetic_bleeding/`
- YOLO model: `artifacts/yolo/bleeding_yolov8n/weights/best.pt`
- YOLO training log: `artifacts/logs/yolo_training.jsonl`

Latest local YOLOv8 validation metrics on the synthetic validation split:

| Metric | Value |
| --- | ---: |
| mAP50 | 0.9950 |
| Precision | 0.9984 |
| Recall | 1.0000 |

## Knowledge Base Build

Build the KB text files and Chroma vector store:

```bash
python3 scripts/build_kb.py
```

Artifacts:

- KB documents: `data/kb/`
- Vector store: `artifacts/vectorstore/`

## Structured Patient Records

Synthetic structured records live in `data/patients/`.

Each file uses this schema:

```json
{
  "patient_id": "P-0001",
  "demographics": {"age": 54, "sex": "female"},
  "chief_complaint": "Palpable breast lump in left breast",
  "history": {
    "personal_history": ["dense breasts"],
    "family_history": ["mother with breast cancer at 62"],
    "comorbidities": ["hypertension"]
  },
  "current_medications": ["losartan 50 mg daily"],
  "recent_exams": {
    "mammography": "BI-RADS 4 lesion in left breast",
    "ultrasound": "solid irregular hypoechoic nodule, 1.8 cm"
  },
  "recent_labs": {"cbc": "within normal limits", "cmp": "within normal limits"},
  "imaging_summary": "Suspicious left breast lesion with recommendation for tissue diagnosis",
  "clinician_notes": "Patient reports lump noticed 3 weeks ago; no fever; mild local tenderness.",
  "ml_context": {
    "malignancy_probability": 0.81,
    "model_label": "Malignant",
    "model_name": "RF"
  },
  "last_updated": "2026-03-23"
}
```

## Run the Assistant

KB-only mode:

```bash
python3 scripts/run_assistant.py
```

Patient-context mode:

```bash
python3 scripts/run_assistant.py --patient-id P-0001
```

Multimodal mode:

```bash
python3 scripts/run_assistant.py \
  --patient-id P-0001 \
  --audio path/to/consultation.wav \
  --video path/to/procedure.mp4
```

Legacy ML-context mode without patient record:

```bash
python3 scripts/run_assistant.py --features "17.99,10.38,122.8,1001,..."
```

Expected behavior:

- The assistant uses the local fine-tuned adapter by default.
- If `ASSISTANT_ALLOW_FALLBACK=false`, adapter load failure returns an error instead of silently switching models.
- If `ASSISTANT_ALLOW_FALLBACK=true`, Gemini may be used only when the local model cannot initialize.
- Final answers include:
  - `Clinical Context Source: patient_record:<id>` when patient context is used
  - `Knowledge Base Sources: ...` when KB docs are used
  - `Audio Source: ...` when audio context is used
  - `Video Source: ...` when video context is used

## Testing

Run the test suite:

```bash
python3 -m pytest -q
```

Recommended smoke checks:

```bash
python3 -m pytest -q tests/test_guardrails.py tests/test_patient_store.py tests/test_audit_logger.py
python3 -m pytest -q tests/test_assistant_graph.py
python3 scripts/generate_synthetic_data.py --count 10 --output-dir /tmp/synthetic_bleeding_smoke
```

## Logging and Audit

Structured logs are written to `artifacts/logs/`.

Key files:

- `artifacts/logs/finetune.jsonl`
- `artifacts/logs/assistant.jsonl`
- `artifacts/logs/audit.jsonl`
- `artifacts/logs/yolo_training.jsonl`

Audit entries include:

- `user_query`
- `retrieved_docs`
- `model_response`
- `guardrail_triggered`
- `patient_id`
- `patient_context_used`
- `kb_sources`
- `patient_source`

## Data Provenance

Phase 4 uses public and synthetic data to simulate the challenge requirement for internal medical data:

- MedQuAD
- PubMedQA
- Synthetic oncology instruction pairs
- Synthetic structured patient records
- Synthetic laparoscopic-style frames for anomalous bleeding detection

This repository does not contain real hospital PHI.

## Report and Demo

Phase 3 report:

- [Relatorio_Tecnico_Tech_Challenge_Fase3.md](/Users/theonetto/www/pos-ia-dev/fiap-ia-challenge-phase-2/docs/reports/Relatorio_Tecnico_Tech_Challenge_Fase3.md)

Phase 4 report:

- [Relatorio_Tecnico_Tech_Challenge_Fase4.md](/Users/theonetto/www/pos-ia-dev/fiap-ia-challenge-phase-2/docs/reports/Relatorio_Tecnico_Tech_Challenge_Fase4.md)

Phase 4 demo video:

- TODO: add final YouTube or Vimeo link before delivery.
