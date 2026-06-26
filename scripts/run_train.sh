#!/usr/bin/env bash
# Example training launcher. Requires a CUDA GPU (~16 GB VRAM, e.g. T4 / A10).
#
# Usage:
#   export HF_TOKEN=hf_xxx        # your Hugging Face token (gated Mistral model)
#   bash scripts/run_train.sh
set -euo pipefail

# Load .env if present so HF_TOKEN is available.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export it or add it to .env (see .env.example)." >&2
  exit 1
fi

python -m src.train --config config.yaml

# After training, optionally merge the adapter into the base model:
#   python -m src.merge --config config.yaml
