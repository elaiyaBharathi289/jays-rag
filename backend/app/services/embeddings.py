from functools import lru_cache
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


@lru_cache
def get_local_embedder():
    settings = get_settings()

    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_document_embedder():
    return get_local_embedder()


def get_query_embedder():
    return get_local_embedder()


def get_embedder():
    return get_local_embedder()


async def embed_documents(
    texts: list[str],
    batch_size: int = 8,
) -> list[list[float]]:

    embedder = get_local_embedder()

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        print(
            f"Embedding batch: "
            f"{start + 1}-{min(start + batch_size, len(texts))} "
            f"of {len(texts)} chunks"
        )

        embeddings = await asyncio.to_thread(
            embedder.embed_documents,
            batch,
        )

        all_embeddings.extend(embeddings)

    return all_embeddings


async def embed_query(text: str) -> list[float]:

    embedder = get_local_embedder()

    return await asyncio.to_thread(
        embedder.embed_query,
        text,
    )