import hashlib
import time
from functools import lru_cache
from pathlib import Path
from langchain_chroma import Chroma
from app.core.config import get_settings
from app.services.embeddings import get_query_embedder

settings = get_settings()

@lru_cache
def get_cache_store():
    path = Path(settings.chroma_path) / "semantic_cache"
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="semantic_answer_cache",
        embedding_function=get_query_embedder(),
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
    )

def _key(question: str, document_id: str | None) -> str:
    raw = f"{document_id or '*'}::{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()

async def lookup(question: str, document_id: str | None):
    if not settings.semantic_cache_enabled:
        return None
    store = get_cache_store()
    try:
        results = await store.asimilarity_search_with_relevance_scores(
            question, k=settings.semantic_cache_top_k,
            filter={"document_id": document_id or "*"},
        )
    except Exception:
        return None
    if not results:
        return None
    doc, score = results[0]
    created = float(doc.metadata.get("created_at", 0))
    if created and time.time() - created > settings.semantic_cache_ttl_seconds:
        return None
    if score >= settings.semantic_cache_threshold:
        return {"answer": doc.page_content, "score": score}
    return None

async def put(question: str, document_id: str | None, answer: str):
    if not settings.semantic_cache_enabled:
        return
    store = get_cache_store()
    try:
        await store.aadd_texts(
            texts=[answer],
            metadatas=[{"document_id": document_id or "*", "created_at": time.time()}],
            ids=[_key(question, document_id)],
        )
    except Exception:
        # Cache must never make the RAG request fail.
        return
