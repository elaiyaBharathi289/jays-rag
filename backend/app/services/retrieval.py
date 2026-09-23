from app.db.database import connect
from app.services.vector_store import dense_search
from app.services.llm import transform_query, rerank_chunks
from app.core.config import get_settings


settings = get_settings()


# ============================================================
# RRF SCORE
# ============================================================

def rrf_score(rank: int) -> float:
    """
    Calculate Reciprocal Rank Fusion score.

    Rank starts at 1.
    """

    return 1.0 / (
        settings.rrf_k + rank
    )


# ============================================================
# BM25 SEARCH
# ============================================================

async def bm25_search(
    query: str,
    top_k: int,
    document_id: str | None = None,
):
    """
    Retrieve chunks using SQLite FTS5 + BM25.
    """

    db = await connect()

    try:

        # ----------------------------------------------------
        # Convert query into safe words
        # ----------------------------------------------------

        words = [
            word.strip()
            for word in query.split()
            if word.strip()
        ]

        if not words:
            return []

        safe_words = []

        for word in words:

            cleaned = "".join(
                char
                for char in word
                if char.isalnum()
                or char == "_"
            )

            if cleaned:
                safe_words.append(cleaned)

        if not safe_words:
            return []

        # ----------------------------------------------------
        # Build FTS5 query
        #
        # Example:
        #
        # company leave policy
        #
        # becomes:
        #
        # "company" AND "leave" AND "policy"
        # ----------------------------------------------------

        fts_query = " AND ".join(
            f'"{word}"'
            for word in safe_words
        )

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        sql = """
        SELECT
            c.id,
            c.document_id,
            c.text,
            c.page_number,
            c.section,
            bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        JOIN chunks c
            ON c.id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
        """

        params = [
            fts_query
        ]

        # ----------------------------------------------------
        # Restrict to selected document
        # ----------------------------------------------------

        if document_id:

            sql += """
            AND c.document_id = ?
            """

            params.append(
                document_id
            )

        # ----------------------------------------------------
        # Ranking + limit
        # ----------------------------------------------------

        sql += """
        ORDER BY bm25_score
        LIMIT ?
        """

        params.append(
            top_k
        )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        cursor = await db.execute(
            sql,
            tuple(params),
        )

        rows = await cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        await db.close()


# ============================================================
# DENSE SEARCH
# ============================================================

async def dense_search_candidates(
    query: str,
    top_k: int,
    document_id: str | None = None,
):
    """
    Convert raw Chroma response into
    simple candidate dictionaries.
    """

    results = await dense_search(
        query=query,
        n_results=top_k,
        document_id=document_id,
    )

    ids = results.get(
        "ids",
        [[]],
    )[0]

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    candidates = []

    for i in range(len(ids)):

        metadata = (
            metadatas[i]
            or {}
        )

        candidates.append(
            {
                "id": ids[i],
                "document_id": metadata.get(
                    "document_id"
                ),
                "text": documents[i],
                "page_number": metadata.get(
                    "page_number"
                ),
                "section": metadata.get(
                    "section",
                    "",
                ),
                "dense_score": float(
                    distances[i]
                ),
            }
        )

    return candidates


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

async def hybrid_retrieve(
    question: str,
    history: list[dict],
    document_id: str | None,
):
    """
    Complete hybrid retrieval pipeline.

    Question
        ↓
    Query transformation
        ↓
    Original + rewritten query
        ↓
    ┌───────────────────┐
    │                   │
    ↓                   ↓
    BM25              Dense
    SQLite            Chroma
    │                   │
    └─────────┬─────────┘
              ↓
             RRF
              ↓
          Candidates
              ↓
          Reranking
              ↓
        Final chunks
    """

    # ========================================================
    # 1. QUERY TRANSFORMATION
    # ========================================================

    rewritten = await transform_query(
        question,
        history,
    )

    # Use original + rewritten query.
    queries = list(
        dict.fromkeys(
            [
                question,
                rewritten,
            ]
        )
    )

    # ========================================================
    # 2. FUSED RESULTS
    # ========================================================

    fused: dict[str, dict] = {}

    # ========================================================
    # 3. SEARCH BOTH RETRIEVERS
    # ========================================================

    for query in queries:

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        try:

            bm25_results = await bm25_search(
                query=query,
                top_k=settings.bm25_top_k,
                document_id=document_id,
            )

            print(
                f"BM25 found "
                f"{len(bm25_results)} chunks "
                f"for query: {query}"
            )

        except Exception as exc:

            print(
                f"BM25 retrieval failed: {exc}"
            )

            bm25_results = []

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        try:

            dense_results = (
                await dense_search_candidates(
                    query=query,
                    top_k=settings.dense_top_k,
                    document_id=document_id,
                )
            )

            print(
                f"Dense retrieval found "
                f"{len(dense_results)} chunks "
                f"for query: {query}"
            )

        except Exception as exc:

            print(
                f"Dense retrieval failed: {exc}"
            )

            dense_results = []

        # ====================================================
        # 4. ADD BM25 RESULTS TO RRF
        # ====================================================

        for rank, row in enumerate(
            bm25_results,
            start=1,
        ):

            chunk_id = row["id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    **row,
                    "rrf_score": 0.0,
                }

            fused[chunk_id][
                "rrf_score"
            ] += rrf_score(rank)

        # ====================================================
        # 5. ADD DENSE RESULTS TO RRF
        # ====================================================

        for rank, row in enumerate(
            dense_results,
            start=1,
        ):

            chunk_id = row["id"]

            if not chunk_id:
                continue

            if chunk_id not in fused:

                fused[chunk_id] = {
                    **row,
                    "rrf_score": 0.0,
                }

            else:

                fused[chunk_id].update(
                    {
                        key: value
                        for key, value in row.items()
                        if value is not None
                    }
                )

            fused[chunk_id][
                "rrf_score"
            ] += rrf_score(rank)

    # ========================================================
    # 6. SORT BY RRF
    # ========================================================

    candidates = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    # Retrieve more candidates than final context.
    candidates = candidates[
        : settings.rerank_top_k * 3
    ]

    print(
        f"Hybrid retrieval produced "
        f"{len(candidates)} candidates."
    )

    # ========================================================
    # 7. RERANK
    # ========================================================

    reranked = await rerank_chunks(
        rewritten,
        candidates,
    )

    # ========================================================
    # 8. FINAL CHUNKS
    # ========================================================

    final_chunks = reranked[
        : settings.rerank_top_k
    ]

    print(
        f"Reranker selected "
        f"{len(final_chunks)} chunks."
    )

    return (
        rewritten,
        final_chunks,
    )