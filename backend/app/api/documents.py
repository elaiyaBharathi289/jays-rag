import uuid

from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    BackgroundTasks,
)

from app.core.config import get_settings
from app.core.errors import AppError

from app.services.ingestion import (
    sha256_file,
    create_document_record,
    process_document,
)

from app.schemas.api import UploadResponse


router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
)

settings = get_settings()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # ---------------------------------------------------------
    # 1. Validate filename
    # ---------------------------------------------------------

    if not file.filename:
        raise AppError(
            "Filename is required.",
            400,
            "INVALID_FILENAME",
        )

    # ---------------------------------------------------------
    # 2. Validate file type
    # ---------------------------------------------------------

    suffix = Path(
        file.filename
    ).suffix.lower()

    if suffix not in {
        ".pdf",
        ".docx",
        ".txt",
    }:
        raise AppError(
            "Only PDF, DOCX and TXT files are supported.",
            415,
            "UNSUPPORTED_FILE",
        )

    # ---------------------------------------------------------
    # 3. Create storage path
    # ---------------------------------------------------------

    file_path = (
        Path(settings.upload_dir)
        / f"{uuid.uuid4()}{suffix}"
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    size = 0

    try:

        # -----------------------------------------------------
        # 4. Save uploaded file
        # -----------------------------------------------------

        with file_path.open("wb") as out:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(chunk)

                if size > settings.max_upload_bytes:

                    raise AppError(
                        f"File exceeds "
                        f"{settings.max_upload_mb} MB limit.",
                        413,
                        "FILE_TOO_LARGE",
                    )

                out.write(chunk)

        # -----------------------------------------------------
        # 5. Calculate SHA-256
        # -----------------------------------------------------

        digest = sha256_file(file_path)

        # -----------------------------------------------------
        # 6. Create / update document record
        # -----------------------------------------------------

        record, duplicate = await create_document_record(
            filename=file.filename,
            sha256=digest,
            mime_type=file.content_type,
            size=size,
            file_path=file_path,
        )

        # -----------------------------------------------------
        # 7. Existing non-failed document
        # -----------------------------------------------------

        if duplicate:

            file_path.unlink(
                missing_ok=True
            )

            return UploadResponse(
                document_id=record["id"],
                filename=file.filename,
                status=record["status"],
                duplicate=True,
                message="This document was already uploaded.",
            )

        # -----------------------------------------------------
        # 8. Start ingestion in background
        # ---------------------------------------------------------

        background_tasks.add_task(
            process_document,
            record["id"],
            file_path,
        )

        # -----------------------------------------------------
        # 9. Return queued response
        # -----------------------------------------------------

        return UploadResponse(
            document_id=record["id"],
            filename=file.filename,
            status="queued",
            message="Document accepted and ingestion started.",
        )

    except Exception:

        # If the upload itself fails before the document
        # is successfully registered, clean up the file.

        file_path.unlink(
            missing_ok=True
        )

        raise


@router.get("")
async def list_documents():

    from app.db.database import connect

    db = await connect()

    try:

        cursor = await db.execute(
            """
            SELECT
                id,
                filename,
                size_bytes,
                file_path,
                status,
                error_message,
                created_at,
                updated_at
            FROM documents
            ORDER BY created_at DESC
            """
        )

        documents = [
            dict(row)
            for row in await cursor.fetchall()
        ]

        return {
            "documents": documents
        }

    finally:

        await db.close()