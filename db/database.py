"""
database.py — SQLite persistence layer for the RAG support assistant.

Stores:
  - messages    : every chat turn (user + assistant) with confidence + sources
  - escalations : HITL review records (replaces the old in-memory manager)
  - feedback    : thumbs up/down ratings on assistant answers

Uses stdlib sqlite3 with WAL mode. Thread-safe via a connection-per-call
pattern suitable for Streamlit's multi-threaded execution.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app.config import CHROMA_DB_PATH

# Persist DB alongside the vector store data
_DB_DIR  = os.path.dirname(os.path.abspath(CHROMA_DB_PATH))
_DB_PATH = os.path.join(_DB_DIR, "assistant.db")


@contextmanager
def _conn():
    """Yield a SQLite connection with row access by column name."""
    con = sqlite3.connect(_DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    """Create tables if they don't exist. Call once at app startup."""
    with _conn() as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT,
                role             TEXT NOT NULL,
                content          TEXT NOT NULL,
                confidence       TEXT,
                similarity_pct   REAL,
                sources          TEXT,
                created_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS escalations (
                thread_id        TEXT PRIMARY KEY,
                query            TEXT NOT NULL,
                ai_answer        TEXT,
                confidence       TEXT,
                reason           TEXT,
                status           TEXT NOT NULL DEFAULT 'PENDING',
                final_answer     TEXT,
                overridden       INTEGER DEFAULT 0,
                created_at       TEXT NOT NULL,
                resolved_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id       INTEGER,
                session_id       TEXT,
                query            TEXT,
                rating           TEXT NOT NULL,
                comment          TEXT,
                created_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_rating  ON feedback(rating);
            """
        )


def _now() -> str:
    return datetime.now().isoformat()


# ── Messages ──────────────────────────────────────────────────────────────────
def add_message(session_id: str, role: str, content: str,
                confidence: str | None = None, similarity_pct: float | None = None,
                sources: list | None = None) -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO messages (session_id, role, content, confidence, "
            "similarity_pct, sources, created_at) VALUES (?,?,?,?,?,?,?)",
            (session_id, role, content, confidence, similarity_pct,
             json.dumps(sources or []), _now()),
        )
        return cur.lastrowid


def get_messages(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id ASC", (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_messages(session_id: str):
    with _conn() as con:
        con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))


# ── Escalations ───────────────────────────────────────────────────────────────
def create_escalation(thread_id: str, query: str, ai_answer: str,
                      confidence: str, reason: str):
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO escalations (thread_id, query, ai_answer, "
            "confidence, reason, status, created_at) VALUES (?,?,?,?,?,'PENDING',?)",
            (thread_id, query, ai_answer, confidence, reason, _now()),
        )


def resolve_escalation(thread_id: str, final_answer: str, overridden: bool):
    with _conn() as con:
        con.execute(
            "UPDATE escalations SET status=?, final_answer=?, overridden=?, "
            "resolved_at=? WHERE thread_id=?",
            ("OVERRIDDEN" if overridden else "APPROVED", final_answer,
             1 if overridden else 0, _now(), thread_id),
        )


def get_escalations(limit: int = 100) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM escalations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_escalations() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM escalations WHERE status='PENDING' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Feedback ──────────────────────────────────────────────────────────────────
def add_feedback(rating: str, query: str = "", comment: str = "",
                 message_id: int | None = None, session_id: str | None = None):
    with _conn() as con:
        con.execute(
            "INSERT INTO feedback (message_id, session_id, query, rating, comment, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (message_id, session_id, query, rating, comment, _now()),
        )


def get_feedback(limit: int = 500) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Analytics queries ─────────────────────────────────────────────────────────
def get_stats() -> dict:
    with _conn() as con:
        total_msgs = con.execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'"
        ).fetchone()[0]
        total_esc  = con.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
        conf_rows  = con.execute(
            "SELECT confidence, COUNT(*) c FROM messages "
            "WHERE role='assistant' AND confidence IS NOT NULL GROUP BY confidence"
        ).fetchall()
        fb_rows = con.execute(
            "SELECT rating, COUNT(*) c FROM feedback GROUP BY rating"
        ).fetchall()

    return {
        "total_queries"    : total_msgs,
        "total_escalations": total_esc,
        "confidence_dist"  : {r["confidence"]: r["c"] for r in conf_rows},
        "feedback_dist"    : {r["rating"]: r["c"] for r in fb_rows},
    }


def get_queries_per_day(days: int = 14) -> dict:
    """Return {date_str: count} of user queries over the last N days."""
    with _conn() as con:
        rows = con.execute(
            "SELECT substr(created_at, 1, 10) d, COUNT(*) c FROM messages "
            "WHERE role='user' GROUP BY d ORDER BY d DESC LIMIT ?", (days,)
        ).fetchall()
    return {r["d"]: r["c"] for r in reversed(rows)}


def get_top_queries(limit: int = 10) -> list[dict]:
    """Most frequent user questions."""
    with _conn() as con:
        rows = con.execute(
            "SELECT content AS query, COUNT(*) c FROM messages "
            "WHERE role='user' GROUP BY content ORDER BY c DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_document_usage(limit: int = 20) -> list[dict]:
    """
    How often each source document appears in assistant answers.
    Parses the JSON sources column of assistant messages.
    """
    counts: dict[str, int] = {}
    with _conn() as con:
        rows = con.execute(
            "SELECT sources FROM messages WHERE role='assistant' AND sources IS NOT NULL"
        ).fetchall()
    for r in rows:
        try:
            for s in json.loads(r["sources"] or "[]"):
                name = s.get("source") or s.get("filename") or s.get("collection") or "unknown"
                counts[name] = counts.get(name, 0) + 1
        except Exception:
            continue
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"document": k, "citations": v} for k, v in ordered]


def get_escalation_rate() -> float:
    """Fraction of user queries that were escalated to a human."""
    with _conn() as con:
        q = con.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0]
        e = con.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    return (e / q) if q else 0.0


def get_knowledge_gaps(limit: int = 50) -> list[dict]:
    """
    Questions the system struggled with: low-confidence answers or
    answers that received negative feedback. Surfaces topics HR/KB
    owners should document.
    """
    with _conn() as con:
        # For each low-confidence ASSISTANT answer, fetch the preceding USER question
        # from the same session (matched by the closest prior message id).
        low_conf = con.execute(
            """
            SELECT
              (SELECT u.content FROM messages u
                 WHERE u.session_id = m.session_id AND u.role='user'
                   AND u.id < m.id ORDER BY u.id DESC LIMIT 1) AS query,
              m.confidence, m.similarity_pct, m.created_at
            FROM messages m
            WHERE m.role='assistant' AND m.confidence='LOW'
            ORDER BY m.created_at DESC LIMIT ?
            """, (limit,)
        ).fetchall()
        neg_fb = con.execute(
            "SELECT query, NULL AS confidence, NULL AS similarity_pct, created_at "
            "FROM feedback WHERE rating='down' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    rows = [dict(r) for r in low_conf] + [dict(r) for r in neg_fb]
    return rows[:limit]
