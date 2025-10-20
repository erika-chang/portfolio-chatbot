# Builds a FAISS index from all documents in data/source/ and subfolders (PDF/MD/TXT)
import os, json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from src.config import Config

DATA_DIR = Path("data")
SRC_DIR = DATA_DIR / "source"
IDX_DIR = DATA_DIR / "index"

_embed = SentenceTransformer(Config.EMBEDDING_MODEL)

def _pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def _load_text(path: Path) -> str:
    return _pdf_to_text(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")

def _split(text: str) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(i + Config.CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - Config.CHUNK_OVERLAP)
    return chunks

def main():
    # You can force a specific file via DOC_FILENAME env
    target = os.getenv("DOC_FILENAME")
    if target:
        src = SRC_DIR / target
        if not src.exists():
            raise SystemExit(f"File not found: {src}")
        files = [src]
    else:
        files = [p for p in SRC_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt"}]
        if not files:
            raise SystemExit("No documents found in data/source/ (PDF/MD/TXT)")

    all_chunks = []
    all_metas = []
    all_vecs_list = []

    for file in files:
        text = _load_text(file)
        chunks = _split(text)
        vecs = _embed.encode(chunks, normalize_embeddings=True)
        vecs = np.array(vecs, dtype="float32")
        all_vecs_list.append(vecs)
        all_chunks.extend(chunks)
        source_name = str(file.relative_to(SRC_DIR))
        all_metas.extend([{"source": source_name, "text": c} for c in chunks])

    if all_vecs_list:
        all_vecs = np.concatenate(all_vecs_list, axis=0)
        index = faiss.IndexFlatIP(all_vecs.shape[1])
        index.add(all_vecs)

        IDX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(IDX_DIR / "faiss.index"))

        with open(IDX_DIR / "meta.json", "w", encoding="utf-8") as f:
            json.dump(all_metas, f, ensure_ascii=False)

        print(f"Indexed {len(all_chunks)} chunks from {len(files)} files")
    else:
        print("No chunks to index")

if __name__ == "__main__":
    main()
