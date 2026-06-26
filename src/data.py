"""Data loading + prompt formatting for the medical Q&A dataset.

The dataset is a list of {instruction, input, output} records (ChatDoctor-style
doctor/patient pairs). We format each record into an instruction-tuning prompt and
expose a train/val split for SFTTrainer.

No secrets, no hardcoded paths here - the dataset path comes from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# System-prompt prefix injected so the model (and anyone reading a transcript) is
# reminded this is a demo, not clinical advice.
SAFETY_SYSTEM_PREFIX = (
    "You are a medical Q&A assistant for an educational demo. "
    "You are NOT a licensed clinician and your answers are NOT medical advice. "
    "Always remind the user to consult a qualified healthcare professional."
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_training_prompt(sample: dict[str, str]) -> str:
    """Format one record into a Mistral instruction prompt WITH the target response.

    Used by SFTTrainer's ``formatting_func`` during fine-tuning. We use Mistral's
    [INST] ... [/INST] chat format so the fine-tuned adapter is consistent with the
    instruct base model's expected template.
    """
    instruction = sample["instruction"].strip()
    user_input = sample["input"].strip()
    response = sample["output"].strip()

    user_message = f"{instruction}\n\nPatient: {user_input}"
    # Mistral instruct format: <s>[INST] {user} [/INST] {assistant}</s>
    return f"<s>[INST] {user_message} [/INST] Doctor: {response}</s>"


def build_inference_prompt(question: str, instruction: str | None = None) -> str:
    """Format a single patient question into a Mistral prompt WITHOUT the answer.

    Used at generation time. The model continues after ``[/INST] Doctor:``.
    """
    if instruction is None:
        instruction = (
            "If you are a doctor, please answer the medical questions "
            "based on the patient's description."
        )
    user_message = f"{instruction}\n\nPatient: {question.strip()}"
    return f"<s>[INST] {user_message} [/INST] Doctor:"


def load_dataset_splits(config: dict[str, Any]):
    """Load the JSON dataset and return a (train, val) tuple of HF Datasets.

    Imported lazily so that modules which only need the prompt helpers (e.g. the
    FastAPI inference layer) don't require ``datasets`` to be installed.
    """
    from datasets import load_dataset  # local import keeps light deps optional

    data_cfg = config["data"]
    dataset_path = data_cfg["dataset_path"]
    if not Path(dataset_path).is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set data.dataset_path in config.yaml."
        )

    full = load_dataset("json", data_files={"all": dataset_path})["all"]
    n = len(full)
    val_size = int(data_cfg.get("val_size", 29))
    val_size = max(1, min(val_size, n - 1))

    train_ds = full.select(range(0, n - val_size))
    val_ds = full.select(range(n - val_size, n))
    return train_ds, val_ds
