from dotenv import load_dotenv
load_dotenv()  # make .env available project-wide

import os

class Config:
    # LLM (Mistral)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral")
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral-large-latest")
    LLM_API_KEY = os.getenv("MISTRAL_API_KEY") or os.getenv("LLM_API_KEY")

    # RAG knobs
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 60))
    TOP_K = int(os.getenv("TOP_K", 8))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.4))
