"""Merge the trained LoRA adapter into the base model and save a standalone model.

Merging removes the runtime PEFT dependency at inference time and produces a plain
transformers model directory that can be loaded directly (or pushed to the Hub).

Note: merging loads the base model in full precision (fp16/bf16), so this step needs
more host/GPU memory than 4-bit inference. The HF token is read from HF_TOKEN.

Run:
    python -m src.merge --config config.yaml
"""

from __future__ import annotations

import argparse
import os

from .data import load_config


def main(config_path: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set. Export your Hugging Face token (see .env.example)."
        )

    base_model_id = config["model"]["base_model_id"]
    adapter_dir = config["training"]["output_dir"]
    merged_dir = config["merge"]["merged_output_dir"]

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(
            f"Adapter directory '{adapter_dir}' not found. Run `python -m src.train` first."
        )

    print(f"Loading base model (fp16) for merge: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )

    print(f"Attaching adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.save_pretrained(merged_dir)

    print(f"Merged model saved to: {merged_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
