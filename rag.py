import json, re
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import Config
from src.llm.mistral_service import MistralLLMService

DATA_DIR = "data/index"
INDEX_PATH = f"{DATA_DIR}/faiss.index"
META_PATH  = f"{DATA_DIR}/meta.json"


_embed = SentenceTransformer(Config.EMBEDDING_MODEL)
_llm   = MistralLLMService()
_index = None
_meta  = None

# rag.py (top of file)
_embed = None
_llm = None
_index = None
_meta  = None

def _ensure_models():
    global _embed, _llm
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        from src.config import Config
        _embed = SentenceTransformer(Config.EMBEDDING_MODEL)
    if _llm is None:
        from src.llm.mistral_service import MistralLLMService
        _llm = MistralLLMService()

def _ensure_index():
    import faiss, json
    from pathlib import Path
    global _index, _meta
    if _index is None:
        p = Path("data/index/faiss.index")
        m = Path("data/index/meta.json")
        if not p.exists() or not m.exists():
            # Graceful: return 0 hits if index missing
            _index = None
            _meta = []
            return
        _index = faiss.read_index(str(p))
        _meta  = json.load(m.open(encoding="utf-8"))

def _load_store():
    global _index, _meta
    if _index is None:
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)

def _emb(q: List[str]):
    v = _embed.encode(q, normalize_embeddings=True)
    return np.array(v, dtype="float32")

# Lightweight language guess (en/pt/nl) so English-in → English-out, etc.
def _guess_lang(text: str) -> str:
    t = text.lower()
    pt_markers = ["você", "vc ", "quê", "qual", "quais", "onde", "quando", "por que", "porque", "não", "obrigado", "obrigada", "com", "para", "sobre"]
    nl_markers = ["waar", "hoe", "wat", "wanneer", "welke", "jij", "je ", "niet", "alstublieft", "dank", "met", "naar", "over"]
    if any(m in t for m in pt_markers) or re.search(r"[áàâãéêíóôõúç]", t):
        return "pt"
    if any(m in t for m in nl_markers):
        return "nl"
    return "en"

def _lang_name(code: str) -> str:
    return {"en": "English", "pt": "Portuguese", "nl": "Dutch"}.get(code, "English")

BASE_SYSTEM_PROMPT = (
    "You are Erika's friendly portfolio assistant 🤖💫.\n"
    "Use only the provided context; if information is missing, reply: 'I don't know based on the current document.'\n"
    "Keep answers short (1–3 sentences). If a list is requested, use up to 3 concise bullets.\n"
    "Add 1–2 tasteful emojis when appropriate (e.g., 😊💡📊✨🎯). Never invent facts."
)

def retrieve(question: str) -> List[dict]:
    _load_store()
    q = _emb([question])
    faiss.normalize_L2(q)
    scores, idxs = _index.search(q, Config.TOP_K)
    out = []
    for s, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        doc = _meta[i]
        out.append({
            "text": doc["text"],
            "source": doc.get("source", "document")
        })
    return out

def build_context(snips: List[dict]) -> str:
    return "\n\n---\n\n".join([f"[{s['source']}] {s['text']}" for s in snips])

def _distinct_sources(hits: List[dict], limit: int = 3) -> List[str]:
    seen, cites = set(), []
    for h in hits:
        src = h.get("source", "document")
        if src not in seen:
            seen.add(src)
            cites.append(src)
        if len(cites) >= limit:
            break
    return cites

async def answer(question: str) -> Tuple[str, List[dict]]:
    hits = retrieve(question)
    code = _guess_lang(question)

    if not hits:
        msg = {
            "en": "I don't know based on the current document 🤷‍♀️",
            "pt": "Não sei com base no documento atual 🤷‍♀️",
            "nl": "Ik weet het niet op basis van het huidige document 🤷‍♀️",
        }.get(code, "I don't know based on the current document 🤷‍♀️")
        return (msg, [])

    ctx = build_context(hits)
    target_language = _lang_name(code)

    system_prompt = (
        BASE_SYSTEM_PROMPT
        + f"\n\nIMPORTANT: Always respond in {target_language} regardless of the context language."
    )

    prompt = (
        "Task: Provide a friendly, natural answer using ONLY the information below. "
        "If a list is requested, use up to 3 concise bullets. "
        "Add 1–2 relevant emojis, but don't overdo it.\n"
        f"Question: {question}\n\nContext:\n{ctx}"
    )

    text = await _llm.generate_response(
        prompt,
        system=system_prompt,
        temperature=max(Config.TEMPERATURE, 0.4),
        max_tokens=400
    )

    citations = [{"source": s} for s in _distinct_sources(hits)]
    return text, citations
