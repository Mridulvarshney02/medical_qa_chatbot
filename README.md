# Medical Q&A Chatbot — QLoRA fine-tuning of Mistral-7B

> ⚠️ **SAFETY / MEDICAL DISCLAIMER**
> **Educational/demo only — not medical advice, not a substitute for a licensed clinician.**
> This project fine-tunes a small model on a tiny dataset and can produce confidently wrong,
> incomplete, or unsafe answers. Do **not** use it for diagnosis, treatment, dosing, or any
> real clinical decision. In an emergency, contact your local emergency services.

Fine-tune `mistralai/Mistral-7B-Instruct` into a doctor-style medical Q&A assistant using
**QLoRA** (4-bit NF4 quantization + LoRA adapters), then serve it behind a **FastAPI** API
with a **Streamlit** chat UI.

---

## Problem

Patients ask free-text health questions; we want a model that responds in a concise,
doctor-style register grounded in a curated set of doctor/patient exchanges — while being
honest that it is a demo, not a clinician.

## Approach — QLoRA on Mistral-7B

- **Base model:** `Mistral-7B-Instruct` loaded in **4-bit** with bitsandbytes
  (`nf4` quant type, double quantization, **bfloat16** compute dtype).
- **Adapter:** LoRA (`r=64`, `alpha=16`, `dropout=0.1`) on the attention + MLP projections
  (`q/k/v/o_proj`, `gate/up/down_proj`).
- **Trainer:** TRL `SFTTrainer` with a `formatting_func` that wraps each record in Mistral's
  `<s>[INST] ... [/INST] Doctor: ...</s>` chat template.
- **Serve:** load base + adapter (4-bit) or a **merged** standalone model; generate with the
  chat template; every answer is suffixed with the safety disclaimer.

This is the QLoRA pattern: freeze a 4-bit-quantized base and train only small LoRA matrices,
so a 7B model fine-tunes on a single ~16 GB GPU.

## Dataset

`data/medical_dataset.json` — **329** ChatDoctor-style `{instruction, input, output}`
doctor/patient pairs. The last 29 records are held out for eval (300 train / 29 val).

> **Honest note:** 329 examples is *tiny* — this is a teaching/demo dataset. Expect the
> adapter to nudge tone/format more than to add reliable medical knowledge.

## Architecture (text diagram)

```
                 data/medical_dataset.json (329 pairs)
                              │
                  src/data.py │  load + [INST] prompt format + split
                              ▼
   ┌─────────────────────── TRAINING (GPU) ───────────────────────┐
   │  src/train.py: 4-bit NF4 base ─► LoRA ─► TRL SFTTrainer       │
   │                         │                                      │
   │                         ▼                                      │
   │             adapters/mistral-medical-qa/  (LoRA weights)       │
   │                         │                                      │
   │  src/merge.py (optional): merge adapter ─► models/...-merged/  │
   └───────────────────────────────────────────────────────────────┘
                              │
                src/inference.py │  load (merged | base+adapter), generate()
                              ▼
        app/api.py (FastAPI POST /chat)  ◄────  app/ui.py (Streamlit chat)
```

## Tech stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `accelerate` · `datasets` · `torch`
(CUDA) · `FastAPI` · `uvicorn` · `Streamlit` · `pyyaml` · `python-dotenv`

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs a CUDA torch build for training/4-bit
cp .env.example .env                      # then put your HF_TOKEN in .env
```

`Mistral-7B-Instruct` is a **gated** model: accept the license on its Hugging Face page and
create a read token at https://huggingface.co/settings/tokens. The token is read from the
`HF_TOKEN` environment variable — it is **never** hardcoded.

## How to train (requires a GPU)

**GPU requirement:** ~16 GB VRAM (e.g. NVIDIA **T4** or **A10**). On a T4, set `bf16: false`
and `fp16: true` in `config.yaml` (T4 has no bf16). All hyperparameters live in `config.yaml`.

```bash
export HF_TOKEN=hf_xxx        # or rely on .env
bash scripts/run_train.sh     # python -m src.train --config config.yaml
# saves the LoRA adapter to adapters/mistral-medical-qa/
```

Optionally merge the adapter into a standalone model:

```bash
python -m src.merge --config config.yaml   # -> models/mistral-medical-qa-merged/
```

## How to serve

```bash
# 1) API (loads the model once at startup)
uvicorn app.api:app --host 0.0.0.0 --port 8000

curl -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"I have had a headache for 3 days, what should I do?"}'

# 2) Chat UI (talks to the API by default; MODE=local loads in-process)
streamlit run app/ui.py
```

> The example request/response above is **illustrative**. Training requires a GPU and was
> **not executed in this repository's authoring environment**, so no checkpoint is shipped
> and any numbers/answers shown are not real model outputs.

## Limitations & safety

- **Not a medical device.** Outputs may be wrong, outdated, biased, or unsafe.
- **Tiny dataset (329 pairs)** → limited coverage; the adapter mostly shapes tone/format.
- **No grounding / no citations** — answers are generated, not retrieved from verified sources.
- **No PII / safety filtering** on inputs or outputs beyond the disclaimer prefix/suffix.
- The safety disclaimer is injected as a system-prompt prefix, appended to every answer, and
  shown as a UI banner — but it does **not** make the model safe for real use.

## Repository layout

```
medical-qa-chatbot/
├── config.yaml            # model id, LoRA params, training args, paths
├── data/medical_dataset.json
├── src/
│   ├── data.py            # load + [INST] prompt template + split
│   ├── train.py           # QLoRA fine-tune (4-bit, LoRA, SFTTrainer)
│   ├── merge.py           # merge adapter into base
│   └── inference.py       # load + generate() with safety disclaimer
├── app/
│   ├── api.py             # FastAPI POST /chat
│   └── ui.py              # Streamlit chat UI
└── scripts/run_train.sh
```

## Provenance

The QLoRA/PEFT/TRL setup was extracted and cleaned from a graduate LLMs lab notebook. A
hardcoded Hugging Face token and a hardcoded local file path in the original notebook were
**removed**; secrets now come only from the environment and all paths from `config.yaml`.
