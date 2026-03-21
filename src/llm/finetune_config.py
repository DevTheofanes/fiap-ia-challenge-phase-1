from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FinetuneConfig:
    # Model — LLaMA as suggested by challenge spec
    base_model_name: str = "meta-llama/Llama-3.2-1B"

    # LoRA (per M2 spec: r=8, lora_alpha=16)
    # q_proj/v_proj are the standard attention targets for LLaMA architecture
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_seq_length: int = 256
    gradient_accumulation_steps: int = 4

    # Checkpointing
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True

    # Reproducibility
    seed: int = 42

    # Text formatting — instruction format standard for LLaMA
    text_field: str = "text"
    prompt_template: str = "### Question: {prompt}\n### Answer: {completion}"
