"""QLoRA fine-tuning of Mistral-7B-Instruct on the medical Q&A dataset.

Pipeline:
  1. Load the base model in 4-bit (NF4 + double quant, bfloat16 compute).
  2. Prepare for k-bit training and attach a LoRA adapter.
  3. Fine-tune with TRL's SFTTrainer using the [INST] prompt template.
  4. Save the LoRA adapter to ``training.output_dir``.

Requires a CUDA GPU (~16 GB VRAM, e.g. T4 / A10). The HF token is read from the
HF_TOKEN environment variable - NEVER hardcode it. All hyperparameters come from
config.yaml.

Run:
    python -m src.train --config config.yaml
"""

from __future__ import annotations

import argparse
import os

from .data import build_training_prompt, load_config, load_dataset_splits


def _resolve_compute_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(name, torch.bfloat16)


def main(config_path: str) -> None:
    # Heavy imports kept inside main so `--help` / py_compile don't need torch installed.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    config = load_config(config_path)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Mistral is a gated model. "
            "Export your Hugging Face token: `export HF_TOKEN=...` (see .env.example)."
        )

    model_cfg = config["model"]
    base_model_id = model_cfg["base_model_id"]
    tokenizer_id = model_cfg.get("tokenizer_id") or base_model_id

    # --- 4-bit (QLoRA) quantization config ---
    q = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=_resolve_compute_dtype(
            q.get("bnb_4bit_compute_dtype", "bfloat16")
        ),
    )

    print(f"Loading base model in 4-bit: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=False,
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- LoRA adapter ---
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    train_ds, val_ds = load_dataset_splits(config)
    print(f"Train examples: {len(train_ds)} | Eval examples: {len(val_ds)}")

    def formatting_func(batch):
        # SFTTrainer passes a batch dict of columns; return a list of formatted strings.
        instructions = batch["instruction"]
        inputs = batch["input"]
        outputs = batch["output"]
        return [
            build_training_prompt(
                {"instruction": ins, "input": inp, "output": out}
            )
            for ins, inp, out in zip(instructions, inputs, outputs)
        ]

    # --- Training args (TRL SFTConfig subclasses TrainingArguments) ---
    t = config["training"]
    use_bf16 = bool(t.get("bf16", True)) and torch.cuda.is_available()
    use_fp16 = bool(t.get("fp16", False)) and not use_bf16

    sft_config = SFTConfig(
        output_dir=t["output_dir"],
        max_steps=t.get("max_steps", -1),
        num_train_epochs=t.get("num_train_epochs", 1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "constant"),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=t.get("eval_steps", 20),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 50),
        max_seq_length=t.get("max_seq_length", 1024),
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=t.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    out_dir = t["output_dir"]
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved LoRA adapter + tokenizer to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Mistral-7B on medical Q&A.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
