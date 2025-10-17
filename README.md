# RAG Portfolio Bot (Mistral)

One-document RAG API for your portfolio website. Uses FAISS + SentenceTransformers for retrieval and Mistral for generation.

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add MISTRAL_API_KEY
# put your CV or about_me.md into data/source/
python ingest.py
uvicorn app:app --reload --port 8000
