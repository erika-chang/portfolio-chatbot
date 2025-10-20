# tests/test_grounding_behavior.py
import requests
from .conftest import idk_phrase
from .conftest import idk_phrase, post_with_retry

def test_when_no_sources_answer_should_acknowledge_limits(api_url):
    # Pergunta específica fora do seu corpus padrão
    question = "What is the capital of Planet Vulcan?"
    r = requests.post(api_url, json={"question": question}, timeout=20)
    r.raise_for_status()
    data = r.json()
    # Se a API não retornar fontes, esperamos uma resposta cautelosa
    if not data.get("sources"):
        assert idk_phrase(data["answer"])

def test_when_no_sources_answer_should_acknowledge_limits(api_url):
    question = "What is the capital of Planet Vulcan?"
    r = post_with_retry(api_url, json_payload={"question": question})
    r.raise_for_status()
    data = r.json()
    if not data.get("sources"):
        assert idk_phrase(data["answer"])
