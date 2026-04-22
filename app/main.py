"""
main.py — RAG Customer Support Assistant — CLI Entry Point

Runs an interactive loop that:
  1. Accepts a user query
  2. Passes it through the full LangGraph pipeline
     (retrieval → generation → evaluation → HITL if needed)
  3. Prints the final answer and pipeline metadata
  4. Exits on "exit" or "quit"

Run with:
    python app/main.py
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # Load .env before importing anything that uses API keys

from graph.workflow import run_rag_pipeline


# ─── Visual helpers ───────────────────────────────────────────────────────────

_DIVIDER  = "=" * 60
_SUBDIV   = "-" * 60
_BANNER   = """
╔══════════════════════════════════════════════════════════╗
║       RAG Customer Support Assistant  (Gemini + ChromaDB)║
║       Type  'exit'  or  'quit'  to stop.                 ║
╚══════════════════════════════════════════════════════════╝
"""


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    print(_BANNER)

    while True:
        try:
            # ── Prompt ──────────────────────────────────────────────────────
            query = input("You: ").strip()

            # Exit commands
            if query.lower() in ("exit", "quit", "q"):
                print("\n  Goodbye! Shutting down the assistant.\n")
                break

            # Skip empty input
            if not query:
                print("  (Please enter a question.)\n")
                continue

            print(_SUBDIV)

            # ── Run the full pipeline ────────────────────────────────────────
            result = run_rag_pipeline(query)

            # ── Display result ───────────────────────────────────────────────
            print(_DIVIDER)

            answer       = result.get("answer", "No answer generated.")
            hitl_needed  = result.get("hitl_needed", False)
            scores       = result.get("scores", [])
            avg_score    = round(sum(scores) / len(scores), 4) if scores else "N/A"
            is_confident = result.get("is_confident", False)

            print(f"\n  Answer:\n  {answer}\n")
            print(_SUBDIV)
            print(f"  Avg Relevance Score : {avg_score}  (lower = more relevant)")
            print(f"  Confident           : {is_confident}")
            print(f"  HITL Triggered      : {'Yes — human response used' if hitl_needed else 'No'}")
            print(_DIVIDER + "\n")

        except KeyboardInterrupt:
            # Graceful Ctrl+C exit
            print("\n\n  Interrupted. Goodbye!\n")
            break

        except Exception as e:
            # Surface errors without crashing the loop
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            print("  Please try again.\n")


if __name__ == "__main__":
    main()
