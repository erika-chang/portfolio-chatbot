# rag.py (top of file)
_embed = None
_llm = None
_index = None
_meta  = None

def _ensure_models():
    global _embed, _llm
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        from src.config import Config
        _embed = SentenceTransformer(Config.EMBEDDING_MODEL)
    if _llm is None:
        from src.llm.mistral_service import MistralLLMService
        _llm = MistralLLMService()

def _ensure_index():
    import faiss, json
    from pathlib import Path
    global _index, _meta
    if _index is None:
        p = Path("data/index/faiss.index")
        m = Path("data/index/meta.json")
        if not p.exists() or not m.exists():
            # Graceful: return 0 hits if index missing
            _index = None
            _meta = []
            return
        _index = faiss.read_index(str(p))
        _meta  = json.load(m.open(encoding="utf-8"))
