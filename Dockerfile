FROM python:3.11-slim
WORKDIR /app

# FAISS needs this
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Cloud Run injects $PORT
ENV PORT=8000

# On startup: if index missing, try to ingest; then start Uvicorn
CMD ["bash", "-lc", "python - <<'PY'\nimport os, pathlib, subprocess, sys\np = pathlib.Path('data/index/faiss.index')\nif not p.exists():\n  print('[startup] No FAISS index found. Ingesting...')\n  try:\n    subprocess.check_call([sys.executable, 'ingest.py'])\n    print('[startup] Ingest OK')\n  except Exception as e:\n    print('[startup] Ingest failed:', e)\nimport uvicorn\nuvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT',8000)))\nPY"]
