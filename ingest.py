# ingest.py
import os, json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

SRC_DIR = DATA_DIR / "source"
IDX_DIR = DATA_DIR / "index"
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))


_embed = SentenceTransformer(EMBED_MODEL_NAME)



def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "".join([(p.extract_text() or "") for p in reader.pages])


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    return path.read_text(encoding="utf-8")


def _split(text: str) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - CHUNK_OVERLAP)
    return chunks




def main():
    files = [p for p in SRC_DIR.glob("*") if p.suffix.lower() in {".pdf", ".md", ".txt"}]
    if not files:
        raise SystemExit("Put your single document in data/source (PDF/MD/TXT)")
    # pick the first file
    src = files[0]
    text = _load_text(src)
    chunks = _split(text)
    vecs = _embed.encode(chunks, normalize_embeddings=True)
    vecs = np.array(vecs, dtype="float32")


index = faiss.IndexFlatIP(vecs.shape[1])
faiss.normalize_L2(vecs)
index.add(vecs)


IDX_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(IDX_DIR / "faiss.index"))


meta: List[Dict] = [{"source": src.name, "text": c} for c in chunks]
with open(IDX_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)


print(f"Indexed {len(chunks)} chunks from {src.name}")


if __name__ == "__main__":
    main()
