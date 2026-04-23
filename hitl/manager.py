"""
manager.py — HITL Thread Manager.

Manages active HITL sessions: stores pending escalations,
maps thread_ids, and provides status checking.

In a real MNC deployment this would connect to:
  - ServiceNow for ticket creation
  - Slack for agent notifications  
  - Email for escalation alerts
"""
import uuid
from datetime import datetime
from typing import Optional


class HITLManager:
    """Manages pending HITL escalation sessions."""

    def __init__(self):
        # thread_id → escalation record
        self._pending: dict[str, dict] = {}

    def create_session(self, query: str, ai_answer: str,
                       confidence: str, reason: str) -> str:
        """
        Create a new HITL escalation session.
        Returns a unique thread_id for this session.
        """
        thread_id = f"thread_{uuid.uuid4().hex[:12]}"
        self._pending[thread_id] = {
            "thread_id"   : thread_id,
            "query"       : query,
            "ai_answer"   : ai_answer,
            "confidence"  : confidence,
            "reason"      : reason,
            "status"      : "PENDING",    # PENDING | APPROVED | OVERRIDDEN
            "created_at"  : datetime.now().isoformat(),
            "resolved_at" : None,
            "final_answer": None,
        }
        return thread_id

    def resolve(self, thread_id: str, final_answer: str, overridden: bool):
        """Mark a HITL session as resolved."""
        if thread_id in self._pending:
            self._pending[thread_id]["status"]       = "OVERRIDDEN" if overridden else "APPROVED"
            self._pending[thread_id]["final_answer"] = final_answer
            self._pending[thread_id]["resolved_at"]  = datetime.now().isoformat()

    def get_session(self, thread_id: str) -> Optional[dict]:
        return self._pending.get(thread_id)

    def get_all_pending(self) -> list[dict]:
        return [s for s in self._pending.values() if s["status"] == "PENDING"]

    def get_history(self) -> list[dict]:
        return list(self._pending.values())


# Module-level singleton
hitl_manager = HITLManager()
