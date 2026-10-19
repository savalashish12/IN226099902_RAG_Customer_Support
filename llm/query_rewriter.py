"""
query_rewriter.py — Conversational query rewriting for multi-turn RAG.

Given recent chat history and the user's latest message, produces a
self-contained "standalone" query suitable for retrieval. This lets the
user ask follow-ups like "what about for managers?" without restating
the full context.

Falls back to the original query if rewriting fails or is unnecessary.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# Pronouns / fragments that usually signal a context-dependent follow-up
_FOLLOWUP_MARKERS = (
    "it", "that", "this", "they", "them", "those", "these", "he", "she",
    "what about", "and ", "also", "how about", "why", "which one", "more",
)


def _needs_rewrite(query: str, history: list[dict]) -> bool:
    """Heuristic: rewrite only when there's history and the query is context-dependent."""
    if not history:
        return False
    q = query.lower().strip()
    if len(q.split()) < 4:
        return True
    return any(q.startswith(m) or f" {m} " in f" {q} " for m in _FOLLOWUP_MARKERS)


def rewrite_query(query: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up question into a standalone query using chat history.

    Args:
        query: The user's latest message
        history: Recent turns as [{"role": "user"|"assistant", "content": str}, ...]

    Returns:
        A standalone query string (original query if no rewrite needed/failed).
    """
    if not _needs_rewrite(query, history):
        return query

    # Keep only the last few turns, trimmed, for a compact prompt
    recent = history[-6:]
    convo = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content'][:400]}" for turn in recent
    )

    prompt = (
        "You rewrite follow-up questions into standalone search queries.\n"
        "Using the conversation, rewrite the user's latest message so it is "
        "self-contained and understandable without the conversation.\n"
        "Output ONLY the rewritten query text — no preamble, no quotes.\n"
        "If the latest message is already standalone, output it unchanged.\n\n"
        f"<conversation>\n{convo}\n</conversation>\n\n"
        f"<latest_message>\n{query}\n</latest_message>"
    )

    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        return query
