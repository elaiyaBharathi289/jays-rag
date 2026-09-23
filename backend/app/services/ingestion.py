import hashlib
import uuid

from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError
from app.db.database import connect
from app.services.extraction import extract_file
from app.services.chunking import build_chunks
from app.services.vector_store import add_chunks


settings = get_settings()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    """

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


async def create_document_record(
    filename,
    sha256,
    mime_type,
    size,
    file_path,
):
    """
    Create a document record.

    Behavior:
    - indexed/processing/queued document -> duplicate
    - failed document -> reuse its ID and update its file path
    - new document -> create a new record
    """

    db = await connect()

    try:
        cur = await db.execute(
            """
            SELECT
                id,
                status,
                filename,
                file_path
            FROM documents
            WHERE sha256 = ?
            """,
            (sha256,),
        )

        existing = await cur.fetchone()

        # ---------------------------------------------------------
        # Existing document
        # ---------------------------------------------------------

        if existing:
            existing_record = dict(existing)

            # Previous ingestion failed.
            # Allow the document to be uploaded again.
            if existing_record["status"] == "failed":

                await db.execute(
                    """
                    UPDATE documents
                    SET
                        filename = ?,
                        mime_type = ?,
                        size_bytes = ?,
                        file_path = ?,
                        status = 'queued',
                        error_message = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        filename,
                        mime_type,
                        size,
                        str(file_path),
                        now(),
                        existing_record["id"],
                    ),
                )

                await db.commit()

                return {
                    "id": existing_record["id"],
                    "status": "queued",
                    "filename": filename,
                }, False

            # Already indexed / processing / queued
            return existing_record, True

        # ---------------------------------------------------------
        # New document
        # ---------------------------------------------------------

        doc_id = str(uuid.uuid4())

        await db.execute(
            """
            INSERT INTO documents(
                id,
                filename,
                sha256,
                mime_type,
                size_bytes,
                file_path,
                status,
                error_message,
                created_at,
                updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                doc_id,
                filename,
                sha256,
                mime_type,
                size,
                str(file_path),
                "queued",
                None,
                now(),
                now(),
            ),
        )

        await db.commit()

        return {
            "id": doc_id,
            "status": "queued",
            "filename": filename,
        }, False

    finally:
        await db.close()


async def process_document(
    document_id: str,
    path: Path,
):
    """
    Complete document ingestion pipeline:

    File
      ↓
    Extraction
      ↓
    Chunking
      ↓
    Embedding
      ↓
    Chroma
      ↓
    SQLite chunks
      ↓
    Indexed
    """

    db = await connect()

    try:

        # ---------------------------------------------------------
        # 1. Mark document as processing
        # ---------------------------------------------------------

        print("========================================")
        print(f"Starting ingestion: {document_id}")
        print(f"File: {path}")
        print("========================================")

        await db.execute(
            """
            UPDATE documents
            SET
                status = 'processing',
                updated_at = ?
            WHERE id = ?
            """,
            (
                now(),
                document_id,
            ),
        )

        await db.commit()

        print("✅ Document status: processing")

        # ---------------------------------------------------------
        # 2. Extract document
        # ---------------------------------------------------------

        print("📄 Extracting document...")

        pages = await extract_file(path)

        print(f"✅ Extraction completed: {len(pages)} pages")

        if not pages or not any(
            x["text"].strip()
            for x in pages
        ):
            raise AppError(
                "No extractable text was found in the document.",
                422,
                "EMPTY_DOCUMENT",
            )

        # ---------------------------------------------------------
        # 3. Build chunks
        # ---------------------------------------------------------

        print("✂️ Building chunks...")

        chunks = await build_chunks(pages)

        print(f"✅ Chunking completed: {len(chunks)} chunks")

        if not chunks:
            raise AppError(
                "The document produced no usable chunks.",
                422,
                "NO_CHUNKS",
            )

        # ---------------------------------------------------------
        # 4. Prepare Chroma IDs + metadata
        # ---------------------------------------------------------

        print("🔧 Preparing Chroma data...")

        ids = [
            f"{document_id}:{c.index}"
            for c in chunks
        ]

        texts = [
            c.text
            for c in chunks
        ]

        metadatas = [
            {
                "document_id": document_id,
                "chunk_id": cid,
                "page_number": c.page_number or 0,
                "section": c.section or "",
                "chunk_index": c.index,
                "chunk_type": c.chunk_type,
            }
            for c, cid in zip(chunks, ids)
        ]

        print("✅ Chroma data prepared")

        # ---------------------------------------------------------
        # 5. Generate embeddings + store in Chroma
        # ---------------------------------------------------------

        print("🧠 Starting embeddings + Chroma insertion...")

        await add_chunks(
            ids,
            texts,
            metadatas,
        )

        print("✅ Chroma insertion completed")

        # ---------------------------------------------------------
        # 6. Store chunks in SQLite
        # ---------------------------------------------------------

        print("💾 Starting SQLite chunk insertion...")

        for c, cid in zip(chunks, ids):

            await db.execute(
                """
                INSERT INTO chunks(
                    id,
                    document_id,
                    chunk_index,
                    text,
                    page_number,
                    section,
                    chunk_type,
                    created_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    cid,
                    document_id,
                    c.index,
                    c.text,
                    c.page_number,
                    c.section,
                    c.chunk_type,
                    now(),
                ),
            )

            await db.execute(
                """
                INSERT INTO chunks_fts(
                    chunk_id,
                    document_id,
                    text,
                    section
                )
                VALUES(?,?,?,?)
                """,
                (
                    cid,
                    document_id,
                    c.text,
                    c.section or "",
                ),
            )

        print("✅ SQLite chunks inserted")

        # ---------------------------------------------------------
        # 7. Mark document as indexed
        # ---------------------------------------------------------

        print("📝 Updating document status to indexed...")

        await db.execute(
            """
            UPDATE documents
            SET
                status = 'indexed',
                updated_at = ?,
                error_message = NULL
            WHERE id = ?
            """,
            (
                now(),
                document_id,
            ),
        )

        await db.commit()

        print("========================================")
        print("🎉 DOCUMENT INDEXING COMPLETED")
        print(f"Document ID: {document_id}")
        print("Status: indexed")
        print("========================================")

    except Exception as exc:

        # ---------------------------------------------------------
        # 8. Mark document as failed
        # ---------------------------------------------------------

        print("❌ DOCUMENT INGESTION FAILED")
        print(f"Error: {exc}")

        await db.execute(
            """
            UPDATE documents
            SET
                status = 'failed',
                error_message = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                str(exc)[:1000],
                now(),
                document_id,
            ),
        )

        await db.commit()

    finally:

        # IMPORTANT:
        # Do NOT delete the uploaded file.
        #
        # We keep it so the document can be retried,
        # re-indexed, or processed with a different
        # embedding/chunking strategy later.

        await db.close()

        print("🔒 Database connection closed")