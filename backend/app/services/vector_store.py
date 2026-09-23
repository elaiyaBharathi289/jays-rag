from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.services.embeddings import embed_documents, embed_query


@lru_cache
def get_vector_store():
    settings = get_settings()

    Path(settings.chroma_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name="rag_chunks",
        persist_directory=settings.chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )


async def add_chunks(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
):
    """
    Generate embeddings and store the vectors in Chroma.
    """

    embeddings = await embed_documents(texts)

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(texts)} texts vs "
            f"{len(embeddings)} embeddings"
        )

    store = get_vector_store()

    store._collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )


async def dense_search(
    query: str,
    n_results: int,
    document_id: str | None = None,
):
    """
    Embed the query and perform vector similarity search.
    """

    query_vector = await embed_query(query)

    store = get_vector_store()

    where = None

    if document_id:
        where = {
            "document_id": document_id
        }

    results = store._collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where=where,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    return results


async def delete_document(document_id: str):
    store = get_vector_store()

    store._collection.delete(
        where={"document_id": document_id}
    )