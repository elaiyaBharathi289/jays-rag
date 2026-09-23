import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.errors import AppError
from app.schemas.api import ChatRequest, ChatResponse, Source
from app.services.retrieval import hybrid_retrieve
from app.services.llm import answer_question
from app.services.semantic_cache import (
    lookup as cache_lookup,
    put as cache_put,
)
from app.db.database import connect, get_chat_history
from app.core.config import get_settings


router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)

settings = get_settings()


def now():
    return datetime.now(timezone.utc).isoformat()


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):

    db = await connect()

    try:

        # =====================================================
        # 1. CHECK DOCUMENT
        # =====================================================

        doc = None

        if req.document_id:

            cur = await db.execute(
                """
                SELECT id, filename, status
                FROM documents
                WHERE id=?
                """,
                (req.document_id,),
            )

            doc = await cur.fetchone()

            if not doc:
                raise AppError(
                    "Document not found.",
                    404,
                    "DOCUMENT_NOT_FOUND",
                )

            if doc["status"] != "indexed":
                raise AppError(
                    "Document is not ready yet.",
                    409,
                    "DOCUMENT_NOT_READY",
                )

        # =====================================================
        # 2. CREATE OR LOAD CHAT
        # =====================================================

        chat_id = req.chat_id or str(uuid.uuid4())

        if req.chat_id:

            cur = await db.execute(
                """
                SELECT id
                FROM chats
                WHERE id=?
                """,
                (req.chat_id,),
            )

            if not await cur.fetchone():

                raise AppError(
                    "Chat not found.",
                    404,
                    "CHAT_NOT_FOUND",
                )

        else:

            await db.execute(
                """
                INSERT INTO chats(
                    id,
                    document_id,
                    created_at,
                    updated_at
                )
                VALUES(?,?,?,?)
                """,
                (
                    chat_id,
                    req.document_id,
                    now(),
                    now(),
                ),
            )

            await db.commit()

        # =====================================================
        # 3. LOAD CONVERSATION HISTORY
        # =====================================================

        history = await get_chat_history(chat_id)

        # =====================================================
        # 4. CHECK SEMANTIC CACHE
        # =====================================================

        cached = await cache_lookup(
            req.question,
            req.document_id,
        )

        if cached:

            answer = cached["answer"]

            sources = []

            rewritten = req.question

            cache_hit = True

        else:

            # =================================================
            # 5. HYBRID RETRIEVAL
            # =================================================

            rewritten, candidates = await hybrid_retrieve(
                req.question,
                history,
                req.document_id,
            )

            # =================================================
            # 6. SELECT RERANKED CHUNKS
            # =================================================

            selected = [
                c
                for c in candidates
                if c.get("rerank_score", 0) >= 0.45
            ][:settings.final_context_k]

            # =================================================
            # 7. FALLBACK
            # =================================================

            if not selected:

                selected = candidates[
                    :settings.final_context_k
                ]

            # =================================================
            # 8. NO EVIDENCE
            # =================================================

            if not selected:

                answer = (
                    "I don't know based on the "
                    "uploaded document."
                )

                sources = []

            else:

                # =============================================
                # 9. ENRICH CHUNKS WITH FILENAME
                # =============================================

                enriched = []

                for c in selected:

                    filename = "document"

                    if doc:
                        filename = doc["filename"]

                    else:

                        cur = await db.execute(
                            """
                            SELECT filename
                            FROM documents
                            WHERE id=?
                            """,
                            (c["document_id"],),
                        )

                        row = await cur.fetchone()

                        if row:
                            filename = row["filename"]

                    enriched.append(
                        {
                            **c,
                            "filename": filename,
                        }
                    )

                # =============================================
                # 10. GENERATE FINAL ANSWER
                # =============================================

                answer = await answer_question(
                    req.question,
                    enriched,
                    history,
                )

                # =============================================
                # 11. SAVE TO SEMANTIC CACHE
                # =============================================

                await cache_put(
                    req.question,
                    req.document_id,
                    answer,
                )

                # IMPORTANT:
                # Return enriched sources so the filename
                # appears correctly in the API response.

                sources = enriched

            cache_hit = False

        # =====================================================
        # 12. SAVE USER MESSAGE
        # =====================================================

        await db.execute(
            """
            INSERT INTO messages(
                id,
                chat_id,
                role,
                content,
                created_at
            )
            VALUES(?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                chat_id,
                "user",
                req.question,
                now(),
            ),
        )

        # =====================================================
        # 13. SAVE ASSISTANT MESSAGE
        # =====================================================

        await db.execute(
            """
            INSERT INTO messages(
                id,
                chat_id,
                role,
                content,
                created_at
            )
            VALUES(?,?,?,?,?)
            """,
            (
                str(uuid.uuid4()),
                chat_id,
                "assistant",
                answer,
                now(),
            ),
        )

        # =====================================================
        # 14. UPDATE CHAT
        # =====================================================

        await db.execute(
            """
            UPDATE chats
            SET updated_at=?
            WHERE id=?
            """,
            (
                now(),
                chat_id,
            ),
        )

        await db.commit()

        # =====================================================
        # 15. BUILD SOURCES
        # =====================================================

        source_models = [
            Source(
                document_id=c["document_id"],
                filename=c.get(
                    "filename",
                    "document",
                ),
                page_number=(
                    c.get("page_number")
                    or None
                ),
                chunk_id=c["id"],
                score=float(
                    c.get(
                        "rerank_score",
                        c.get("rrf_score", 0),
                    )
                ),
                text_preview=c["text"][:240],
            )
            for c in sources
        ]

        # =====================================================
        # 16. RETURN RESPONSE
        # =====================================================

        return ChatResponse(
            answer=answer,
            sources=source_models,
            rewritten_query=rewritten,
            chat_id=chat_id,
            cache_hit=cache_hit,
        )

    finally:

        await db.close()