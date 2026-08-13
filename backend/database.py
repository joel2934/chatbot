"""
database.py
Tiny SQLite wrapper. No ORM — plain SQL so it's easy to see exactly
what's happening (this is a learning project, not production infra).
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "chatbot.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'default_user',
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    """)

    conn.commit()
    conn.close()


# ---------- conversations ----------

def create_conversation(title: str = "New Chat", user_id: str = "default_user") -> dict:
    conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversations (conversation_id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, user_id, title, created_at),
    )
    conn.commit()
    conn.close()
    return {"conversation_id": conversation_id, "user_id": user_id, "title": title, "created_at": created_at}


def list_conversations(user_id: str = "default_user") -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_conversation_title(conversation_id: str, title: str):
    conn = get_conn()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE conversation_id = ?", (title, conversation_id)
    )
    conn.commit()
    conn.close()


def search_conversations(query: str, user_id: str = "default_user") -> list:
    """Simple search: matches conversation title OR any message content."""
    conn = get_conn()
    like = f"%{query}%"
    rows = conn.execute(
        """
        SELECT DISTINCT c.* FROM conversations c
        LEFT JOIN messages m ON c.conversation_id = m.conversation_id
        WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
        ORDER BY c.created_at DESC
        """,
        (user_id, like, like),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- messages ----------

def add_message(conversation_id: str, role: str, content: str) -> dict:
    message_id = f"msg_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (message_id, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        (message_id, conversation_id, role, content, timestamp),
    )
    conn.commit()
    conn.close()
    return {"message_id": message_id, "conversation_id": conversation_id, "role": role, "content": content, "timestamp": timestamp}


def get_messages(conversation_id: str, limit: int | None = None) -> list:
    conn = get_conn()
    if limit:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM messages WHERE conversation_id = ?
                ORDER BY timestamp DESC LIMIT ?
            ) sub ORDER BY timestamp ASC
            """,
            (conversation_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conversation_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_conversation(conversation_id: str):
    conn = get_conn()
    conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
    )
    conn.execute(
        "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
    )
    conn.commit()
    conn.close()