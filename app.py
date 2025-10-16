# app.py
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import answer


load_dotenv()


API_KEY = os.getenv("API_AUTH_TOKEN", "CHANGE_ME")
CORS_ALLOW = [s.strip() for s in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")]


app = FastAPI(title="RAG Portfolio Bot — Simple")
app.add_middleware(
CORSMiddleware,
allow_origins=CORS_ALLOW,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"]
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
async def ask(req: AskReq, x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    out, cites = answer(req.question)
    return {"answer": out, "sources": cites}
