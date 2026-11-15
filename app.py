import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/nli-deberta-v3-small"


@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
    mdl.eval()
    return tok, mdl


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def make_chunks(sentences, size=3, step=2):
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), step)]


def check_claim(claim, chunks, tok, mdl):
    inputs = tok(
        chunks, [claim] * len(chunks),
        return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    with torch.no_grad():
        probs = torch.softmax(mdl(**inputs).logits, dim=1)
    labels = [mdl.config.id2label[i].lower() for i in range(probs.shape[1])]
    best_ent = probs[:, labels.index("entailment")].max().item()
    best_con = probs[:, labels.index("contradiction")].max().item()
    if best_ent >= 0.5:
        return "Supported", best_ent
    if best_con >= 0.5:
        return "Contradicted", best_con
    return "Unsupported", 1 - best_ent


st.title("LLM Hallucination Detector")
st.caption("Checks whether each sentence of an AI answer is backed by your source text.")

source = st.text_area("Source text (the evidence)", height=200)
answer = st.text_area("AI answer to check", height=150)

if st.button("Check"):
    if not source.strip() or not answer.strip():
        st.warning("Please fill in both boxes.")
    else:
        with st.spinner("Loading model and checking..."):
            tok, mdl = load_model()
            chunks = make_chunks(split_sentences(source))
            icons = {"Supported": "✅", "Contradicted": "❌", "Unsupported": "⚠️"}
            verdicts = []
            for claim in split_sentences(answer):
                verdict, score = check_claim(claim, chunks, tok, mdl)
                verdicts.append(verdict)
                st.write(f"{icons[verdict]} **{verdict}** ({score:.0%}): {claim}")
        flagged = sum(v != "Supported" for v in verdicts)
        st.metric("Flagged sentences", f"{flagged} / {len(verdicts)}")