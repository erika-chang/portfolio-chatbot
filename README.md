# 🤖 Portfolio RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)]()
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-orange.svg)]()
[![Mistral](https://img.shields.io/badge/LLM-Mistral-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-black.svg)]()

This project is a lightweight **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about my professional background using a **local FAISS index + LLM (Mistral)**. It powers the Q&A feature on my personal portfolio website, enabling fast, contextual and professional responses.

![Demo](docs/demo.gif)

---
## 📌 About
A lightweight, production-ready RAG chatbot designed to power the Q&A section of a portfolio website using FAISS for retrieval and Mistral for generation.

## 🚀 Project Goals

- 📌 Provide an interactive assistant that answers questions about my experience and projects
- 🧠 Retrieve relevant context from my portfolio using **semantic search (FAISS)**
- 💬 Generate concise and accurate responses via **Mistral LLM**
- 🧱 Serve a clean and fast **FastAPI HTTP endpoint**
- 🌐 Allow seamless integration with any frontend (e.g., portfolio website)

---

## 📦 Data Source

**Portfolio & CV (single document)**
Format: `PDF / MD / TXT`
Content: Professional background, experience and project history
Purpose: Serve as the knowledge base for retrieval and generation

---

## 🧱 Project Structure
```bash
portfolio-chatbot/
│
├── data/
│ ├── source/ # Original CV or portfolio document
│ └── index/ # FAISS vector index + metadata
│
├── src/
│ ├── rag/ # Retrieval and answer generation
│ ├── llm/ # Mistral service wrapper
│ └── config.py # Environment variables and settings
│
├── ingest.py # Builds the vector index from the source document
├── app.py # FastAPI app exposing POST /ask
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container for deployment
└── README.md # You're here!
```

---

## 🧰 Tools & Libraries

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| API | FastAPI, Uvicorn, Pydantic |
| Embeddings | SentenceTransformers |
| Vector Store | FAISS |
| LLM | Mistral API |
| PDF Support | PyPDF |
| Config | python-dotenv |
| Deployment | Docker |

---

## 🏗️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-user/portfolio-chatbot.git
cd portfolio-chatbot
```
2️⃣ Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
.\.venv\Scripts\activate     # Windows
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

4️⃣ Create the .env file

```bash
MISTRAL_API_KEY=your_key_here
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLM_MODEL=mistral-small-latest
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
TEMPERATURE=0.3
```

5️⃣ Build the index

```bash
python ingest.py
```

6️⃣ Run the API

```bash
uvicorn app:app --reload
```

7️⃣ Test the endpoint

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are your main tech skills?"}'
```

Example output:

```bash
{
  "answer": "I have experience in Data Science, Python, ML, and LLM-based systems...",
  "sources": ["Erika_CV.pdf"]
}
```

🛠️ Future Enhancements
Add streaming responses

Add confidence scores for context retrieval

Optional authentication and rate limiting
