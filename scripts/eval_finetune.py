"""M2 — Compare base vs fine-tuned LLaMA-3.2-1B using ROUGE metrics.

Usage:
    python scripts/eval_finetune.py                   # all 614 val samples
    python scripts/eval_finetune.py --num-samples 10  # quick check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config
from src.llm.finetune_config import FinetuneConfig
from src.logging_utils import log_event, setup_json_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _generate_answers(model, tokenizer, input_texts: list[str], max_new_tokens: int) -> list[str]:
    import torch
    model.training = False  # disable dropout for deterministic outputs
    results = []
    for text in input_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        # Move inputs to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated tokens (after the input)
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return results


def _rouge_scores(predictions: list[str], references: list[str]) -> dict:
    import evaluate as hf_evaluate
    scorer = hf_evaluate.load("rouge")
    result = scorer.compute(predictions=predictions, references=references)
    return {k: round(float(v), 4) for k, v in result.items()}


def main() -> None:
    args = _parse_args()
    cfg = FinetuneConfig()

    if not config.FINETUNE_FINAL_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter not found at {config.FINETUNE_FINAL_ADAPTER_DIR}. "
            "Run scripts/fine_tune.py first."
        )

    logger = setup_json_logger("finetune", config.FINETUNE_LOG_PATH)
    log_event(logger, "rouge_eval_start", split=args.split, num_samples=args.num_samples)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    data_path   = config.FINETUNE_TRAIN_PATH if args.split == "train" else config.FINETUNE_VAL_PATH
    records     = _load_jsonl(data_path, limit=args.num_samples)
    input_texts = [f"### Question: {r['prompt']}\n### Answer:" for r in records]
    references  = [r["completion"] for r in records]
    log_event(logger, "rouge_eval_data_loaded", n=len(records))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Base model
    log_event(logger, "rouge_eval_base_start")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name, device_map="auto")
    base_preds = _generate_answers(base_model, tokenizer, input_texts, args.max_new_tokens)
    base_rouge = _rouge_scores(base_preds, references)
    base_dur   = round(time.time() - t0, 1)
    log_event(logger, "rouge_eval_base_done", duration_sec=base_dur, **base_rouge)
    del base_model  # free MPS memory

    # Fine-tuned model
    log_event(logger, "rouge_eval_finetuned_start")
    t0 = time.time()
    ft_base  = AutoModelForCausalLM.from_pretrained(cfg.base_model_name, device_map="auto")
    ft_model = PeftModel.from_pretrained(ft_base, str(config.FINETUNE_FINAL_ADAPTER_DIR))
    ft_preds = _generate_answers(ft_model, tokenizer, input_texts, args.max_new_tokens)
    ft_rouge = _rouge_scores(ft_preds, references)
    ft_dur   = round(time.time() - t0, 1)
    log_event(logger, "rouge_eval_finetuned_done", duration_sec=ft_dur, **ft_rouge)

    summary = {
        "split": args.split,
        "n_samples": len(records),
        "base_model": {**base_rouge, "duration_sec": base_dur},
        "finetuned_model": {**ft_rouge, "duration_sec": ft_dur},
        "delta": {k: round(ft_rouge[k] - base_rouge[k], 4) for k in base_rouge},
    }
    log_event(logger, "rouge_eval_summary", **summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
