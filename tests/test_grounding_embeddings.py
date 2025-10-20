# tests/test_grounding_embeddings.py
import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
import requests
from .conftest import load_raw_corpus, idk_phrase

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

@pytest.mark.parametrize("q", [
    "List projects developed",
    "What embedding model is configured?",
])
def test_answer_similarity_to_corpus(q, api_url):
    corpus = load_raw_corpus()
    if not corpus.strip():
        pytest.skip("Corpus vazio em data/source — pule avaliação heurística")
    r = requests.post(api_url, json={"question": q}, timeout=30)
    r.raise_for_status()
    ans = r.json().get("answer","")
    if idk_phrase(ans):
        pytest.skip("Resposta indicou desconhecimento — não avaliar similaridade")
    emb = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    av, cv = emb.encode([ans, corpus], convert_to_numpy=True, normalize_embeddings=True)
    sim = cosine(av, cv)
    # Limiar conservador: >= 0.28 sugere algum alinhamento com a base
    assert sim >= 0.28, f"similaridade baixa com corpus bruto ({sim:.2f}) — possível alucinação"
