"""
notifier.py — Optional outbound notifications for HITL escalations.

If SLACK_WEBHOOK_URL is configured, posts a message when a query is
escalated for human review. Fire-and-forget: failures are swallowed so
notifications never break the pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import urllib.request

from app.config import SLACK_WEBHOOK_URL


def notify_escalation(query: str, ai_answer: str, confidence: str, reason: str) -> bool:
    """
    Post an escalation alert to Slack (if configured).

    Returns True if a notification was sent, False otherwise. Never raises.
    """
    if not SLACK_WEBHOOK_URL:
        return False

    text = (
        "🚨 *New HITL Escalation*\n"
        f"*Query:* {query}\n"
        f"*Confidence:* {confidence}\n"
        f"*Reason:* {reason}\n"
        f"*AI Answer (preview):* {ai_answer[:300]}"
    )

    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
        return True
    except Exception:
        return False
