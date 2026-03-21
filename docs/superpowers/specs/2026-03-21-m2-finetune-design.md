# M2 — Fine-Tuning Pipeline: Design Spec

**Date:** 2026-03-21
**Scope:** Fix TRL 0.29 compatibility in `scripts/fine_tune.py` and define the execution sequence to produce the required M2 artifacts.

---

## Context

The M2 scripts (`fine_tune.py`, `eval_finetune.py`) and config (`src/llm/finetune_config.py`) are fully written but have never been executed. The training dataset is ready (`data/finetune/train.jsonl`, 2 456 pairs; `val.jsonl`, 614 pairs).

The installed environment has TRL 0.29, transformers 5.3, PEFT 0.18, accelerate 1.13 — all significantly newer than the TRL ~0.8 API the script was authored against.

---

## Approach

Minimal targeted fix (Option A): patch only the broken call sites in `fine_tune.py` to match the TRL 0.29 API. No structural redesign, no dep pinning, no changes to `eval_finetune.py` or `FinetuneConfig`.

---

## Code Changes — `scripts/fine_tune.py`

### 1. Switch from `TrainingArguments` to `SFTConfig`

TRL 0.29 consolidates SFT-specific params (`dataset_text_field`, `max_seq_length`, `packing`) into `SFTConfig`, which extends `TrainingArguments`.

```python
# Before
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=...,
    ...  # no dataset_text_field here
)

trainer = SFTTrainer(
    ...
    dataset_text_field=cfg.text_field,
    max_seq_length=cfg.max_seq_length,
    packing=False,
)

# After
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir=...,
    dataset_text_field=cfg.text_field,   # moved here
    max_length=cfg.max_seq_length,       # moved here; param renamed max_seq_length→max_length in TRL 0.29
    packing=False,                        # moved here
    eval_strategy=cfg.evaluation_strategy,  # SFTConfig only accepts eval_strategy, not evaluation_strategy
    ...  # all other TrainingArguments fields unchanged
)

trainer = SFTTrainer(
    ...
    # dataset_text_field, max_seq_length, packing removed from here
)
```

### 2. Pass `eval_strategy` explicitly (not inherited from config attribute name)

`SFTConfig` only accepts `eval_strategy`, not `evaluation_strategy`. Since `fine_tune.py` previously used the kwarg name `eval_strategy=cfg.evaluation_strategy`, this is already correct at the call site — but must be preserved when migrating to `SFTConfig` (do not rely on `**dataclass_fields` expansion).

### 3. Rename `tokenizer` → `processing_class` in `SFTTrainer`

The `tokenizer` parameter was removed from `SFTTrainer` in TRL 0.9+.

```python
# Before
trainer = SFTTrainer(
    ...
    tokenizer=tokenizer,
)

# After
trainer = SFTTrainer(
    ...
    processing_class=tokenizer,
)
```

### Files changed

| File | Change |
|------|--------|
| `scripts/fine_tune.py` | Import `SFTConfig`, replace `TrainingArguments`, rename `tokenizer→processing_class`, move SFT params |
| All other files | No changes |

---

## Execution Sequence

### Prerequisites (one-time, outside codebase)

1. Accept Meta's license at `huggingface.co/meta-llama/Llama-3.2-1B`
2. Set `HF_TOKEN` in `.env`
3. `huggingface-cli login --token $HF_TOKEN`
4. `accelerate config` — select MPS backend

### Run order

```bash
# 1. Smoke test (validates setup before committing ~30 min)
python scripts/fine_tune.py --epochs 1 --batch-size 2

# 2. Full training
python scripts/fine_tune.py   # 3 epochs, batch=4, ~20–30 min on M3 Mac

# 3. Quick eval check
python scripts/eval_finetune.py --num-samples 50

# 4. Full eval
python scripts/eval_finetune.py
```

---

## Expected Artifacts

| Path | Contents |
|------|----------|
| `artifacts/finetune_checkpoints/final_adapter/` | LoRA adapter weights + tokenizer config |
| `artifacts/logs/finetune.jsonl` | Training log (loss, steps, duration) written by `fine_tune.py`; ROUGE eval entries appended by `eval_finetune.py` |

---

## Done Criteria (from milestone spec)

- `scripts/fine_tune.py` executes end-to-end without error
- Checkpoint final saved and loadable with `PeftModel.from_pretrained()`
- Training log recorded in JSONL following `src/logging_utils.py` pattern
- ROUGE comparison (base vs fine-tuned) logged and printed
