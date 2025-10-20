# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pathlib

app = FastAPI(title="RAG Portfolio Bot — Minimal Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    question: str

class AskRes(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskRes)
async def ask(req: AskReq):
    # ⬇️ lazy import so app can start even if RAG is heavy
    from rag import answer
    out, cites = await answer(req.question)
    return {"answer": out, "sources": cites}

@app.get("/_debug_status")
def _debug_status():
    idx = pathlib.Path("data/index/faiss.index").exists()
    meta = pathlib.Path("data/index/meta.json").exists()
    src_files = []
    p = pathlib.Path("data/source")
    if p.exists():
        src_files = [str(x.name) for x in p.iterdir() if x.is_file()]
    return {"index_present": idx, "meta_present": meta, "source_files": src_files}
