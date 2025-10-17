FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/sentence-transformers
RUN python - <<'PY'\nfrom sentence_transformers import SentenceTransformer\nSentenceTransformer('paraphrase-multilingual-mpnet-base-v2')\nprint('Model cached.')\nPY
ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0"]
CMD ["--port", "8080"]
