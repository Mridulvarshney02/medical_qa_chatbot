"""FastAPI service exposing the fine-tuned medical Q&A model.

Endpoints:
  GET  /            -> service banner + safety disclaimer
  GET  /health      -> {"status": "ok", "model_loaded": bool}
  POST /chat        -> {"question": "..."} -> {"answer": "...", "disclaimer": "..."}

The model is loaded ONCE at startup (lifespan) so requests are fast. Config path and
load mode come from env vars CONFIG_PATH / LOAD_MODE. HF_TOKEN is read by the inference
layer from the environment. Run:

    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import SAFETY_DISCLAIMER
from src.inference import MedicalQAModel

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")
LOAD_MODE = os.environ.get("LOAD_MODE")  # None -> auto (merged if present else adapter)

_state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup. If loading fails (e.g. no GPU / no trained model),
    # we keep the service up so /health and / still respond, but /chat will 503.
    try:
        _state["model"] = MedicalQAModel(CONFIG_PATH, load_mode=LOAD_MODE).load()
        print("Model loaded and ready.")
    except Exception as exc:  # noqa: BLE001 - surface as a 503 on /chat
        print(f"WARNING: model failed to load at startup: {exc}")
        _state["model"] = None
    yield
    _state["model"] = None


app = FastAPI(
    title="Medical Q&A Chatbot",
    description=f"QLoRA-tuned Mistral-7B medical Q&A demo. {SAFETY_DISCLAIMER}",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The patient's question.")


class ChatResponse(BaseModel):
    answer: str
    disclaimer: str = SAFETY_DISCLAIMER


@app.get("/")
def root():
    return {
        "service": "medical-qa-chatbot",
        "disclaimer": SAFETY_DISCLAIMER,
        "endpoints": {"health": "GET /health", "chat": "POST /chat"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded (requires a GPU and a trained model/adapter).",
        )
    answer = model.generate(req.question)
    return ChatResponse(answer=answer)
