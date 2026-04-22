"""
escalation.py — Human-in-the-Loop (HITL) escalation for low-confidence queries.

When the evaluation node determines retrieval quality is poor, this node:
  1. Prints the flagged query and AI-generated answer to the console
  2. Asks a human agent to review and provide a better response
  3. Stores the human's response back into state["answer"]

This is CLI-based HITL — simple, practical, and easy to replace
later with a web UI, Slack bot, or ticketing system.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.state import RAGState

# Separator line for visual clarity in CLI
_DIVIDER = "=" * 60


def human_escalation(state: RAGState) -> dict:
    """
    HITL node — pauses the pipeline and requests a human response.

    Called by LangGraph when evaluation_node sets is_confident=False.

    Args:
        state: The current RAGState with query, context, and AI answer.

    Returns:
        Partial state dict with "answer" overwritten by the human's response.
    """
    query      = state["query"]
    ai_answer  = state["answer"]
    scores     = state.get("scores", [])
    avg_score  = round(sum(scores) / len(scores), 4) if scores else "N/A"

    # ── Show human agent full context ────────────────────────────────────────
    print(f"\n{_DIVIDER}")
    print("  !! ESCALATION TRIGGERED — Human Review Required !!")
    print(_DIVIDER)
    print(f"  Query      : {query}")
    print(f"  Avg Score  : {avg_score}  (threshold exceeded — low confidence)")
    print(f"  AI Answer  : {ai_answer}")
    print(_DIVIDER)

    # ── Collect human response ────────────────────────────────────────────────
    print("\n  Please review the query and provide a corrected response.")
    print("  (Press Enter with no text to keep the AI-generated answer)\n")

    human_response = input("  Your Response: ").strip()

    # If agent leaves blank → fall back to AI answer
    if not human_response:
        human_response = ai_answer
        print("\n  [hitl] No input provided. Keeping AI-generated answer.")
    else:
        print("\n  [hitl] Human response recorded.")

    print(_DIVIDER)

    # Return only the fields we're updating — LangGraph merges the rest
    return {
        "answer":       human_response,
        "hitl_needed":  True,   # flag stays True so callers know HITL ran
    }
