# rag.py
import os, json
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

_meta = json.load(f)


def _emb(q: List[str]) -> np.ndarray:
    v = _embed.encode(q, normalize_embeddings=True)
    return np.array(v, dtype="float32")


SYSTEM_PROMPT = (
"You are Erika's portfolio assistant. Answer ONLY from the provided context. "
"If the answer is not in the context, say you don't know. Keep it concise and cite [CV] or [About]."
)


def retrieve(question: str) -> List[dict]:
    _load_store()
    q = _emb([question])
    faiss.normalize_L2(q)
    scores, idxs = _index.search(q, TOP_K)
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        doc = _meta[i]
        out.append({"text": doc["text"], "source": doc["source"], "page": doc.get("page"), "score": float(s)})
    return out




def build_context(snips: List[dict]) -> str:
    parts = []
    for s in snips:
        tag = "[CV]" if s["source"].lower().endswith(".pdf") else "[About]"
        parts.append(f"{tag} {s['text']}")
    return "\n\n".join(parts)

def generate(question: str, context: str) -> str:
    msgs = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Question: {question} Context: {context}")
    ]
    resp = _client.chat(model=LLM_MODEL, messages=msgs, temperature=TEMP, max_tokens=512)
    return resp.choices[0].message.content


def answer(question: str) -> Tuple[str, List[dict]]:
    hits = retrieve(question)
    if not hits:
        return ("I don't know based on the current document.", [])
    ctx = build_context(hits)
    text = generate(question, ctx)
    return text, [{"source": h["source"], "page": h.get("page"), "score": h["score"]} for h in hits]
