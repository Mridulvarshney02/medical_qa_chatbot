"""Streamlit chat UI for the medical Q&A chatbot.

By default it talks to the FastAPI service (POST /chat) so the heavy model lives in one
place. Set MODE=local to load the model in-process instead.

Run:
    streamlit run app/ui.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from src import SAFETY_DISCLAIMER

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODE = os.environ.get("MODE", "api")  # "api" (call FastAPI) or "local" (in-process)

st.set_page_config(page_title="Medical Q&A Chatbot (Demo)", page_icon="🩺")

# Prominent safety banner.
st.warning(f"**SAFETY NOTICE** — {SAFETY_DISCLAIMER}", icon="⚠️")

st.title("Medical Q&A Chatbot")
st.caption("QLoRA fine-tuned Mistral-7B-Instruct · educational demo only")


@st.cache_resource(show_spinner="Loading model into memory...")
def _get_local_model():
    from src.inference import MedicalQAModel

    return MedicalQAModel().load()


def get_answer(question: str) -> str:
    if MODE == "local":
        return _get_local_model().generate(question)
    resp = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=120)
    resp.raise_for_status()
    return resp.json()["answer"]


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(prompt)
            except Exception as exc:  # noqa: BLE001
                answer = f"Error contacting model: {exc}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
