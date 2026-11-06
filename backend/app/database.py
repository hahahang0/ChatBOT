
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "robo_research.db"


@contextmanager
def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    with get_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id);


            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'processing'
                    CHECK(status IN ('processing', 'ready', 'failed')),
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_documents_created
            ON documents(created_at);


            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,

                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,

                UNIQUE(document_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_document_chunks_document
            ON document_chunks(document_id);
        """)


# ============================================================
# CONVERSATION MANAGEMENT
# ============================================================

def create_conversation(conversation_id, title):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO conversations (id, title)
            VALUES (?, ?)
            """,
            (conversation_id, title)
        )


def get_all_conversations():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_conversation(conversation_id):
    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT * FROM conversations
            WHERE id = ?
            """,
            (conversation_id,)
        ).fetchone()

        if conversation is None:
            return None

        messages = connection.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,)
        ).fetchall()

        return {
            "conversation": dict(conversation),
            "messages": [dict(message) for message in messages],
        }


def save_message(conversation_id, role, content):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content)
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (conversation_id,)
        )


def rename_conversation(conversation_id, title):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (title, conversation_id)
        )

        return cursor.rowcount > 0


def delete_conversation(conversation_id):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE id = ?
            """,
            (conversation_id,)
        )

        return cursor.rowcount > 0


# ============================================================
# DOCUMENT MANAGEMENT
# ============================================================

def create_document(
    document_id,
    filename,
    stored_filename,
    file_path,
    file_size,
    file_hash,
):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id,
                filename,
                stored_filename,
                file_path,
                file_size,
                file_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                filename,
                stored_filename,
                file_path,
                file_size,
                file_hash,
            )
        )


def get_document(document_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE id = ?
            """,
            (document_id,)
        ).fetchone()

        return dict(row) if row else None


def get_document_by_hash(file_hash):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,)
        ).fetchone()

        return dict(row) if row else None


def get_all_documents():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                d.file_size,
                d.page_count,
                d.status,
                d.error_message,
                d.created_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c
                ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def update_document_status(
    document_id,
    status,
    page_count=None,
    error_message=None,
):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE documents
            SET
                status = ?,
                page_count = COALESCE(?, page_count),
                error_message = ?
            WHERE id = ?
            """,
            (
                status,
                page_count,
                error_message,
                document_id,
            )
        )


def save_document_chunks(document_id, chunks):
    """
    Save document chunks and their embeddings.

    Each chunk must contain:
        chunk_index
        page_number
        content
        embedding (a list of numbers)
    """

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = ?
            """,
            (document_id,)
        )

        rows = [
            (
                document_id,
                chunk["chunk_index"],
                chunk["page_number"],
                chunk["content"],
                json.dumps(chunk["embedding"]),
            )
            for chunk in chunks
        ]

        connection.executemany(
            """
            INSERT INTO document_chunks (
                document_id,
                chunk_index,
                page_number,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows
        )


def get_document_chunks(document_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,)
        ).fetchall()

        results = []

        for row in rows:
            chunk = dict(row)
            chunk["embedding"] = json.loads(chunk["embedding"])
            results.append(chunk)

        return results


def get_all_document_chunks():
    """
    Return chunks belonging only to successfully processed documents.
    """

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.page_number,
                c.content,
                c.embedding,
                d.filename
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.status = 'ready'
            ORDER BY d.created_at DESC, c.chunk_index ASC
            """
        ).fetchall()

        results = []

        for row in rows:
            chunk = dict(row)
            chunk["embedding"] = json.loads(chunk["embedding"])
            results.append(chunk)

        return results


def delete_document(document_id):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM documents
            WHERE id = ?
            """,
            (document_id,)
        )

        return cursor.rowcount > 0
