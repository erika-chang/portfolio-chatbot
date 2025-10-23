# ========= STAGE 1: builder =========
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Tooling apenas para build de wheels (isolado no builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      git \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Camada cacheável: requirements primeiro
COPY requirements.txt /app/requirements.txt
RUN python -m venv /opt/venv && . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# (Não copiamos o código para o builder — mantemos a imagem final limpa)

# ========= STAGE 2: runtime =========
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/tmp/hf-cache \
    HF_HOME=/tmp/hf-cache

# Somente libs de runtime (inclui wget para o HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      tini \
      wget \
    && rm -rf /var/lib/apt/lists/*

# venv preparada no builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copiar apenas o necessário para rodar
COPY app.py /app/app.py
COPY src/ /app/src/
COPY ingest.py /app/ingest.py

# Garantir diretório do índice (não copiamos data/index do host)
RUN mkdir -p /app/data/index

# Usuário não-root
RUN useradd -u 10001 -m appuser
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/_debug_status || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
