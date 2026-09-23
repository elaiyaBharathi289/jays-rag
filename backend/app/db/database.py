import aiosqlite
from pathlib import Path

from app.core.config import get_settings


settings = get_settings()

DB_PATH = Path(settings.database_url.replace("sqlite:///", ""))

if not DB_PATH.is_absolute():
    DB_PATH = Path(__file__).resolve().parents[2] / DB_PATH


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;


CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    mime_type TEXT,
    size_bytes INTEGER NOT NULL,
    file_path TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_sha256
ON documents(sha256);


CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER,
    section TEXT,
    chunk_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document
ON chunks(document_id);


CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    text,
    section
);


CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    document_id TEXT
        REFERENCES documents(id)
        ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL
        REFERENCES chats(id)
        ON DELETE CASCADE,
    role TEXT NOT NULL
        CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_created
ON messages(chat_id, created_at);
"""


async def init_db():
    """
    Initialize the SQLite database.

    Creates all required tables and indexes.
    Also performs a lightweight migration for existing
    databases by adding the file_path column if necessary.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:

        # Create tables/indexes if they don't exist.
        await db.executescript(SCHEMA)

        # ---------------------------------------------------------
        # Migration:
        # Existing databases were created before file_path existed.
        # Check whether the column already exists.
        # ---------------------------------------------------------

        cursor = await db.execute(
            "PRAGMA table_info(documents)"
        )

        columns = await cursor.fetchall()

        column_names = {
            column[1]
            for column in columns
        }

        if "file_path" not in column_names:
            await db.execute(
                """
                ALTER TABLE documents
                ADD COLUMN file_path TEXT
                """
            )

        await db.commit()


async def connect():
    """
    Create and return a database connection.
    """

    db = await aiosqlite.connect(DB_PATH)

    db.row_factory = aiosqlite.Row

    await db.execute(
        "PRAGMA foreign_keys=ON"
    )

    return db


async def get_chat_history(
    chat_id: str,
    limit: int = 12,
):
    """
    Retrieve the most recent messages for a chat.
    """

    db = await connect()

    try:
        cursor = await db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE chat_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )

        rows = await cursor.fetchall()

        return list(
            reversed(
                [dict(row) for row in rows]
            )
        )

    finally:
        await db.close()