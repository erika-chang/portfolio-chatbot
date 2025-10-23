# Diretórios
DATA_DIR := data
SRC_DIR  := $(DATA_DIR)/source
IDX_DIR  := $(DATA_DIR)/index

# Venv dedicada à INGESTÃO (não usada pela imagem de produção)
ING_VENV := .venv-ingest
PYTHON   := $(ING_VENV)/bin/python
PIP      := $(ING_VENV)/bin/pip

# Arquivo de dependências para ingestão (inclui torch/transformers/sentence-transformers)
ING_REQ  := requirements_ingest.txt

# ---------- Alvos ----------

.PHONY: ingest ingest-clean ingest-venv check-ing-req

# Cria venv de ingestão e instala deps pesadas
ingest-venv: check-ing-req
	python3 -m venv $(ING_VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --no-cache-dir -r $(ING_REQ)

# Gera o índice FAISS fora da imagem (usa ingest.py)
ingest: ingest-venv
	$(PYTHON) ingest.py

# Limpa índice e venv de ingestão (não afeta a imagem)
ingest-clean:
	rm -rf $(IDX_DIR)/*
	rm -rf $(ING_VENV)

# Verificação: exige requirements_ingest.txt
check-ing-req:
	@test -f $(ING_REQ) || (echo "ERRO: $(ING_REQ) não encontrado. Crie-o com deps de ingestão (torch, transformers, sentence-transformers, pypdf, numpy, etc.)."; exit 1)
