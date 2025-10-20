# tests/test_api_contract.py
from .conftest import post_with_retry

def test_contract_minimal_response(api_url):
    r = post_with_retry(api_url, json_payload={"question": "Hello"})
    r.raise_for_status()
    data = r.json()
    assert isinstance(data, dict)
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
