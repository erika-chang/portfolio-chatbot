# ingest.py
# Build a single FAISS index from ALL files under data/source/** (pdf/md/txt)
import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from src.config import Config

DATA_DIR = Path("data")
SRC_DIR  = DATA_DIR / "source"
IDX_DIR  = DATA_DIR / "index"

ALLOWED_EXTS = {".pdf", ".md", ".txt"}

# ---------- helpers ----------
def _pdf_to_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)
    except Exception as e:
        print(f"[warn] PDF read failed: {path.name}: {e}")
        return ""

def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[warn] Text read failed: {path.name}: {e}")
        return ""

def _split_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks

def _iter_source_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    files.sort()
    return files

# ---------- main ----------
def main():
    # load model once
    embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    dim = embed_model.get_sentence_embedding_dimension()

    files = _iter_source_files(SRC_DIR)
    if not files:
        raise SystemExit("Put your files under data/source/ (pdf/md/txt).")

    all_vecs: List[np.ndarray] = []
    meta: List[Dict] = []
    total_chunks = 0
    used_files = 0
    skipped_files = 0

    for f in files:
        text = _load_text(f).strip()
        if not text:
            print(f"[skip] no text extracted: {f.relative_to(SRC_DIR)}")
            skipped_files += 1
            continue
        chunks = _split_words(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
        if not chunks:
            print(f"[skip] empty chunks: {f.relative_to(SRC_DIR)}")
            skipped_files += 1
            continue

        # encode; ensure 2D shape even for single-chunk case
        vecs = embed_model.encode(chunks, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim == 1:
            # single chunk → make it (1, dim)
            vecs = vecs.reshape(1, -1)

        if vecs.size == 0 or vecs.shape[1] != dim:
            # if something odd happened, coerce to empty (0, dim) to keep shapes consistent
            vecs = np.empty((0, dim), dtype="float32")

        if vecs.shape[0] == 0:
            print(f"[skip] produced 0 vectors: {f.relative_to(SRC_DIR)}")
            skipped_files += 1
            continue

        all_vecs.append(vecs)
        for c in chunks:
            meta.append({"source": str(f.relative_to(SRC_DIR)), "text": c})
        total_chunks += len(chunks)
        used_files += 1
        print(f"[ok] {f.relative_to(SRC_DIR)} → {len(chunks)} chunks")

    if not all_vecs:
        raise SystemExit("No vectors generated. Check your files or add a simple about_me.md with plain text.")

    # concat safely: all entries are 2D (n, dim)
    mat = np.concatenate(all_vecs, axis=0)
    faiss.normalize_L2(mat)

    IDX_DIR.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, str(IDX_DIR / "faiss.index"))
    with open(IDX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("\n=== Ingest summary ===")
    print(f"Files scanned   : {len(files)}")
    print(f"Files indexed   : {used_files}")
    print(f"Files skipped   : {skipped_files}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Embedding dim   : {mat.shape[1]}")
    print(f"Index size      : {index.ntotal}")
    print(f"Wrote           : {IDX_DIR/'faiss.index'}, {IDX_DIR/'meta.json'}")

if __name__ == "__main__":
    main()
