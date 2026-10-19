"""
manager.py — HITL Thread Manager (SQLite-persisted).

Manages HITL escalation sessions: records pending escalations, resolves
them with human decisions, and provides an audit log. Records survive
application restarts via the SQLite layer in db/database.py.

In a real MNC deployment this would additionally connect to:
  - ServiceNow for ticket creation
  - Slack for agent notifications
  - Email for escalation alerts
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
from typing import Optional

from db import database as db
from hitl.notifier import notify_escalation


class HITLManager:
    """Manages HITL escalation sessions backed by SQLite."""

    def create_session(self, query: str, ai_answer: str,
                       confidence: str, reason: str) -> str:
        """
        Create a new HITL escalation session.
        Returns a unique thread_id for this session.
        Also fires an optional Slack notification (fire-and-forget).
        """
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        db.create_escalation(thread_id, query, ai_answer, confidence, reason)
        notify_escalation(query, ai_answer, confidence, reason)
        return thread_id

    def resolve(self, thread_id: str, final_answer: str, overridden: bool):
        """Mark a HITL session as resolved (APPROVED or OVERRIDDEN)."""
        db.resolve_escalation(thread_id, final_answer, overridden)

    def get_session(self, thread_id: str) -> Optional[dict]:
        for s in db.get_escalations(limit=1000):
            if s["thread_id"] == thread_id:
                return s
        return None

    def get_all_pending(self) -> list[dict]:
        return db.get_pending_escalations()

    def get_history(self) -> list[dict]:
        """Return escalation history oldest-first (matches prior UI behaviour)."""
        return list(reversed(db.get_escalations(limit=1000)))


# Module-level singleton
hitl_manager = HITLManager()
