"""M2 — Fine-tune meta-llama/Llama-3.2-1B with LoRA on M3 MPS.

Pre-requisites:
    huggingface-cli login --token $HF_TOKEN
    accelerate config   (select MPS once)

Usage:
    python scripts/fine_tune.py                            # full 3-epoch run (~20-30 min M3)
    python scripts/fine_tune.py --epochs 1 --batch-size 2  # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

from src import config
from src.llm.finetune_config import FinetuneConfig
from src.logging_utils import log_event, setup_json_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _load_dataset(path: Path, cfg: FinetuneConfig):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append({
                cfg.text_field: cfg.prompt_template.format(
                    prompt=row["prompt"], completion=row["completion"]
                )
            })
    return Dataset.from_list(records)


def main() -> None:
    args = _parse_args()
    cfg = FinetuneConfig()
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.per_device_train_batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    logger = setup_json_logger("finetune", config.FINETUNE_LOG_PATH)
    log_event(logger, "finetune_start",
              base_model=cfg.base_model_name, lora_r=cfg.lora_r,
              lora_alpha=cfg.lora_alpha, epochs=cfg.num_train_epochs,
              batch_size=cfg.per_device_train_batch_size, lr=cfg.learning_rate)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTConfig, SFTTrainer

    config.FINETUNE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FINETUNE_FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    log_event(logger, "loading_model", model=cfg.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no pad token by default

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        device_map="auto",   # MPS via accelerate
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log_event(logger, "peft_applied", trainable_params=trainable, total_params=total,
              trainable_pct=round(trainable / total * 100, 3))
    model.print_trainable_parameters()

    log_event(logger, "loading_data")
    train_dataset = _load_dataset(config.FINETUNE_TRAIN_PATH, cfg)
    val_dataset   = _load_dataset(config.FINETUNE_VAL_PATH, cfg)
    log_event(logger, "data_loaded", train_size=len(train_dataset), val_size=len(val_dataset))

    training_args = SFTConfig(
        output_dir=str(config.FINETUNE_CHECKPOINTS_DIR),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        seed=cfg.seed,
        report_to="none",
        logging_steps=10,
        fp16=False,
        bf16=False,   # MPS does not support bf16
        # SFT-specific params (moved from SFTTrainer in TRL 0.29)
        dataset_text_field=cfg.text_field,
        max_length=cfg.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field=cfg.text_field,
        max_seq_length=cfg.max_seq_length,
        packing=False,
    )

    log_event(logger, "training_start")
    t0 = time.time()
    train_result = trainer.train()
    log_event(logger, "training_done",
              train_loss=train_result.training_loss,
              steps=train_result.global_step,
              duration_sec=round(time.time() - t0, 1))

    model.save_pretrained(str(config.FINETUNE_FINAL_ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.FINETUNE_FINAL_ADAPTER_DIR))
    log_event(logger, "adapter_saved", path=str(config.FINETUNE_FINAL_ADAPTER_DIR))
    print(f"Done. Adapter saved to {config.FINETUNE_FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
