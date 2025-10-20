FROM python:3.11-slim
WORKDIR /app

# FAISS runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python ingest.py

# (Optional but recommended) cache the SentenceTransformers model at build time
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
# ⬇️ use python -c instead of a heredoc
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); print('Model cached.')"

# Bind to Cloud Run's PORT
ENV PORT=8080
EXPOSE 8080

# Start fast; RAG loads lazily on first request
ENTRYPOINT ["/bin/sh","-lc","exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level info"]
