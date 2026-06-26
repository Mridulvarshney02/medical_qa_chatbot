"""Inference layer: load the fine-tuned model and answer patient questions.

Two load modes:
  - "merged"  : load a standalone merged model from merge.merged_output_dir (default
                if it exists).
  - "adapter" : load the 4-bit base model + LoRA adapter from training.output_dir.

The HF token is read from HF_TOKEN. All paths come from config.yaml. A safety prefix
is baked into every prompt and every answer ends with a disclaimer reminder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import SAFETY_DISCLAIMER
from .data import build_inference_prompt, load_config


class MedicalQAModel:
    """Wraps the tokenizer + model and exposes ``generate(question)``."""

    def __init__(self, config_path: str = "config.yaml", load_mode: str | None = None):
        self.config = load_config(config_path)
        self.load_mode = load_mode or self._auto_mode()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _auto_mode(self) -> str:
        merged_dir = self.config["merge"]["merged_output_dir"]
        if Path(merged_dir).is_dir():
            return "merged"
        return "adapter"

    def load(self) -> "MedicalQAModel":
        """Load tokenizer + model into memory. Call once (e.g. at API startup)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN")  # may be unneeded for local merged dirs
        base_model_id = self.config["model"]["base_model_id"]

        if self.load_mode == "merged":
            merged_dir = self.config["merge"]["merged_output_dir"]
            if not Path(merged_dir).is_dir():
                raise FileNotFoundError(
                    f"Merged model dir '{merged_dir}' not found. "
                    "Run `python -m src.merge` or use load_mode='adapter'."
                )
            print(f"Loading merged model from: {merged_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(merged_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                merged_dir,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            from peft import PeftModel
            from transformers import BitsAndBytesConfig

            if not hf_token:
                raise EnvironmentError(
                    "HF_TOKEN is required to load the gated base model in adapter mode."
                )
            adapter_dir = self.config["training"]["output_dir"]
            if not Path(adapter_dir).is_dir():
                raise FileNotFoundError(
                    f"Adapter dir '{adapter_dir}' not found. Run `python -m src.train` first."
                )
            q = self.config["quantization"]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=q.get("load_in_4bit", True),
                bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print(f"Loading 4-bit base + adapter ({adapter_dir})")
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                device_map="auto",
                quantization_config=bnb_config,
                token=hf_token,
            )
            self.model = PeftModel.from_pretrained(base, adapter_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=hf_token)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._loaded = True
        return self

    def generate(self, question: str) -> str:
        """Generate a doctor-style answer for a patient question."""
        import torch

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        prompt = build_inference_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        inf = self.config["inference"]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=inf.get("max_new_tokens", 512),
                temperature=inf.get("temperature", 0.7),
                top_p=inf.get("top_p", 0.9),
                do_sample=inf.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt).
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return f"{answer}\n\n---\n[Disclaimer] {SAFETY_DISCLAIMER}"


if __name__ == "__main__":
    # Tiny manual smoke test (requires a GPU + trained model).
    qa = MedicalQAModel().load()
    print(qa.generate("I have had a persistent headache for three days. What should I do?"))
