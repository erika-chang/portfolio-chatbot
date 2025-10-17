# Dockerfile (second option variant)
FROM python:3.11-slim
WORKDIR /app

# FAISS runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# (Optional but recommended) cache the model at build so cold start is fast
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
RUN python - <<'PY'\nfrom sentence_transformers import SentenceTransformer\nSentenceTransformer('paraphrase-multilingual-mpnet-base-v2')\nprint('Model cached.')\nPY

# Cloud Run sets PORT (usually 8080). Bind to it explicitly.
ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/bin/sh","-lc","exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
