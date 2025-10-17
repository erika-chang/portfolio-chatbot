from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer

app = FastAPI(title="RAG Portfolio Bot — Minimal Agent")

# Open CORS so your website can call it (restrict to your domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # e.g. ["https://your-site.com"]
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
    out, cites = await answer(req.question)
    return {"answer": out, "sources": cites}
