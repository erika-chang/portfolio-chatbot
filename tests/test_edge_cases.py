# tests/test_edge_cases.py
import pytest
from .conftest import idk_phrase, is_clarification, post_with_retry

@pytest.mark.parametrize("q", ["", "   ", "\n", "??????", "What???   "])
def test_empty_or_vague_returns_idk_or_clarification(api_url, q):
    r = post_with_retry(api_url, json_payload={"question": q})
    r.raise_for_status()
    data = r.json()
    assert "answer" in data and "sources" in data
    a = data["answer"]
    assert idk_phrase(a) or is_clarification(a), f"Esperado 'não sei' OU pedido de clarificação. Resposta: {a!r}"
