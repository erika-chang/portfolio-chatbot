# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pathlib
import os
import json
import threading

app = FastAPI(title="RAG Portfolio Bot — Minimal Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Config & lazy download
# =========================
INDEX_DIR = os.getenv("INDEX_DIR", "/app/data/index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
INDEX_GCS_URI = os.getenv("INDEX_GCS_URI")  # e.g., gs://my-bucket/path/to/index

_download_lock = threading.Lock()
_last_download_err = None  # stored to report in _debug_status


def _parse_gcs_uri(uri: str):
    """
    Parse "gs://bucket/optional/prefix" -> (bucket, prefix or "")
    """
    if not uri or not uri.startswith("gs://"):
        raise ValueError("INDEX_GCS_URI must start with 'gs://'.")
    without = uri[len("gs://") :]
    parts = without.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def _download_from_gcs_if_needed():
    """
    Download faiss.index and meta.json from GCS if missing.
    Keeps the image tiny and sources only in GCS, not baked into the container.
    """
    global _last_download_err
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        return  # already present

    if not INDEX_GCS_URI:
        _last_download_err = "INDEX_GCS_URI is not set. Configure env var with a gs://bucket[/prefix]."
        raise RuntimeError(_last_download_err)

    try:
        from google.cloud import storage  # runtime dependency
    except Exception as e:
        _last_download_err = f"google-cloud-storage not available: {e}"
        raise RuntimeError(_last_download_err)

    bucket_name, prefix = _parse_gcs_uri(INDEX_GCS_URI)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    os.makedirs(INDEX_DIR, exist_ok=True)

    idx_name = f"{prefix}/faiss.index" if prefix else "faiss.index"
    meta_name = f"{prefix}/meta.json" if prefix else "meta.json"

    idx_blob = bucket.blob(idx_name)
    meta_blob = bucket.blob(meta_name)

    if not idx_blob.exists():
        _last_download_err = f"Object not found in GCS: gs://{bucket_name}/{idx_name}"
        raise FileNotFoundError(_last_download_err)
    if not meta_blob.exists():
        _last_download_err = f"Object not found in GCS: gs://{bucket_name}/{meta_name}"
        raise FileNotFoundError(_last_download_err)

    idx_blob.download_to_filename(INDEX_PATH)
    meta_blob.download_to_filename(META_PATH)
    _last_download_err = None  # success


def ensure_index_local():
    """
    Thread-safe: ensures index files exist locally by downloading from GCS when needed.
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        return
    with _download_lock:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            _download_from_gcs_if_needed()


# =========================
# Schemas
# =========================
class AskReq(BaseModel):
    question: str


class AskRes(BaseModel):
    answer: str
    sources: list


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskRes)
async def ask(req: AskReq):
    """
    Loads the FAISS index from GCS on-demand (first request) and delegates to rag.answer.
    The container image does NOT include the index folder; it relies solely on the GCS bucket.
    """
    # Make sure local index files exist (download from GCS if missing)
    try:
        ensure_index_local()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare index from GCS: {e}")

    # Lazy import so startup stays fast
    from rag import answer
    out, cites = await answer(req.question)
    return {"answer": out, "sources": cites}


@app.post("/admin/warmup")
def admin_warmup():
    """
    Optional: proactively download index from GCS.
    """
    try:
        ensure_index_local()
        return {"status": "ok", "downloaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/_debug_status")
def _debug_status():
    idx = pathlib.Path(INDEX_PATH).exists()
    meta = pathlib.Path(META_PATH).exists()
    # Mirror old shape, but include GCS info and any last error for transparency
    return {
        "index_present": idx,
        "meta_present": meta,
        "source_files": [],  # image does not ship sources; data lives in GCS
        "gcs_uri": INDEX_GCS_URI or "",
        "index_dir": INDEX_DIR,
        "last_download_error": _last_download_err,
    }
