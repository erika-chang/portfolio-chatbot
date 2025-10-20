# tests/conftest.py
import os
import re
import time
from typing import Any, Dict, Optional
import requests
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000/ask")
# Permitir timeout maior para cold-start
DEFAULT_TIMEOUT = int(os.getenv("TEST_REQUEST_TIMEOUT", "45"))  # antes: 15

@pytest.fixture(scope="session")
def api_url():
    return API_URL

def looks_like_api_key(text: str) -> bool:
    return bool(re.search(r"\b(sk|rk|pk)_[A-Za-z0-9]{16,}\b", text or ""))

def idk_phrase(answer: str) -> bool:
    a = (answer or "").lower()
    return ("don't know" in a) or ("não sei" in a)

def is_clarification(answer: str) -> bool:
    """
    Detecta respostas de clarificação, que são aceitáveis para entradas vazias/vagas.
    """
    a = (answer or "").lower()
    patterns = [
        r"\bcould you clarify\b",
        r"\bplease clarify\b",
        r"\brefine\b",
        r"\breformule\b",
        r"\bpergunta\b.*\bclara\b",
        r"\bask your question\b",
        r"\bexemplo\b",          # mensagens com exemplos de como perguntar
        r"\bfor example\b",
        r"\bpor exemplo\b",
    ]
    return any(re.search(p, a) for p in patterns)

def post_with_retry(
    url: str,
    json_payload: Dict[str, Any],
    timeout: Optional[int] = None,
    retries: int = 2,
    backoff_sec: float = 5.0,
) -> requests.Response:
    """
    Faz POST com 1-2 tentativas extras para contornar cold start.
    """
    t = timeout or DEFAULT_TIMEOUT
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return requests.post(url, json=json_payload, timeout=t)
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec)
            else:
                raise
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec)
            else:
                raise
    # não deveria chegar aqui
    raise last_exc

def load_raw_corpus() -> str:
    """
    Opcional: usa data/source para heurística de grounding.
    """
    from pathlib import Path
    buf = []
    root = Path("data/source")
    if not root.exists():
        return ""
    for p in root.glob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            buf.append(p.read_text(encoding="utf-8", errors="ignore"))
        elif p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                r = PdfReader(str(p))
                pages = []
                for pg in r.pages:
                    try:
                        pages.append(pg.extract_text() or "")
                    except Exception:
                        pages.append("")
                buf.append("\n\n".join(pages))
            except Exception:
                pass
    return "\n\n".join(buf)
