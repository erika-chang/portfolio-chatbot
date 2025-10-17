# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
