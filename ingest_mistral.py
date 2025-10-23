# ingest_mistral.py
import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List
from mistralai import Mistral
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ========= Config =========
SOURCE_DIR = Path("data/source")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "mistral-embed"  # 1024-dim
DIM = 1024
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY in your environment.")

client = Mistral(api_key=API_KEY)

# ========= Utils =========
def read_file_text(p: Path) -> str:
    if p.suffix.lower() in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix.lower() == ".pdf":
        text = []
        with open(p, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ""
                text.append(t)
        return "\n".join(text)
    # ignore other types
    return ""

def chunk_text(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    s = " ".join(s.split())  # collapse whitespace
    if not s:
        return []
    chunks = []
    i = 0
    while i < len(s):
        chunk = s[i : i + size]
        chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks

def embed_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embeds in batches using the new client: client.embeddings.create(model=..., inputs=[...]).
    Returns float32 array of shape (N, DIM).
    """
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Mistral expects list[str]; ensure all are strings and non-empty
        batch = [t if isinstance(t, str) else str(t) for t in batch]
        batch = [t for t in batch if t.strip()]
        if not batch:
            continue
        res = client.embeddings.create(model=MODEL, inputs=batch)
        # res.data is List[Embedding] with .embedding list[float]
        for item in res.data:
            vecs.append(item.embedding)
    if not vecs:
        return np.zeros((0, DIM), dtype="float32")
    arr = np.array(vecs, dtype="float32")
    # normalize for cosine similarity
    faiss.normalize_L2(arr)
    return arr

# ========= Load docs & build corpus =========
def load_corpus() -> List[str]:
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"Source dir not found: {SOURCE_DIR.resolve()}. Put your files there.")
    texts = []
    for p in sorted(SOURCE_DIR.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        content = read_file_text(p)
        parts = chunk_text(content)
        texts.extend(parts)
    # sanitize: only non-empty strings
    texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise RuntimeError("No text chunks found. Ensure data/source has .txt/.md/.pdf with readable text.")
    return texts

def main():
    texts = load_corpus()
    print(f"Chunks to embed: {len(texts)}")

    embeddings = embed_batch(texts, batch_size=32)
    if embeddings.shape[0] != len(texts):
        raise RuntimeError(f"Embeddings count mismatch: {embeddings.shape[0]} vs {len(texts)}")

    # FAISS index (cosine via inner product + L2-normalized vectors)
    index = faiss.IndexFlatIP(DIM)
    index.add(embeddings)

    # Save outputs
    index_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "meta.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL,
                "dim": DIM,
                "metric": "cosine",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "texts": texts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote index: {index_path}  and meta: {meta_path}")

if __name__ == "__main__":
    main()
