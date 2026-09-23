# DocuRAG

Production-style RAG learning project using HTML/CSS/Bootstrap/JS + FastAPI + Groq + Gemini Embeddings + ChromaDB + SQLite.

## Architecture
- Upload -> SHA-256 dedup -> SQLite document record -> background ingestion
- Extraction: PDF/DOCX/TXT
- Structure-aware chunking + semantic chunking
- Gemini Embedding 2
- Chroma dense vector search
- SQLite FTS5 BM25 lexical search
- Query transformation with Groq
- Reciprocal Rank Fusion (RRF)
- Groq LLM reranking
- Grounded answer generation with an explicit no-hallucination policy
- SQLite chat history

## Run backend
```powershell
cd backend
copy .env.example .env
# put your keys in .env
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run frontend
Use VS Code Live Server on `frontend/index.html`, or any static web server on port 5500.

## Important
This is production-style code for a single-machine project. For a multi-instance deployment, replace FastAPI BackgroundTasks with a durable queue such as Redis + Celery/Arq, and consider PostgreSQL/object storage.
