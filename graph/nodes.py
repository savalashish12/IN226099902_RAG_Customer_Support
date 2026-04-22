"""
nodes.py — LangGraph processing nodes.

Flow: retrieval_node → generation_node → evaluation_node → (END | hitl_node)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import SIMILARITY_THRESHOLD
from retrieval.retriever import retrieve_context_with_scores
from llm.gemini_model import generate_answer
from graph.state import RAGState


# ── Node 1: Retrieval ─────────────────────────────────────────────────────────

def retrieval_node(state: RAGState) -> dict:
    query           = state["query"]
    collection_name = state["collection_name"]

    print(f"\n[retrieval] Query: '{query}' | Collection: '{collection_name}'")

    results  = retrieve_context_with_scores(query, collection_name=collection_name)

    context  = [text  for text, _,      _ in results]
    scores   = [score for _,    score,  _ in results]

    print(f"[retrieval] Got {len(context)} chunks | Scores: {[round(s,3) for s in scores]}")
    for i, c in enumerate(context):
        print(f"  [{i+1}] score={scores[i]:.4f} | {c[:80]}...")

    return {"context": context, "scores": scores}


# ── Node 2: Generation ────────────────────────────────────────────────────────

def generation_node(state: RAGState) -> dict:
    query   = state["query"]
    context = state["context"]

    print(f"\n[generation] Generating answer...")
    answer = generate_answer(query, context)
    print(f"[generation] Answer: {answer[:120]}...")

    return {"answer": answer}


# ── Node 3: Evaluation ────────────────────────────────────────────────────────

def evaluation_node(state: RAGState) -> dict:
    scores = state.get("scores", [])

    if not scores:
        print("[evaluation] No scores — triggering HITL.")
        return {"is_confident": False, "hitl_needed": True}

    avg_score    = sum(scores) / len(scores)
    is_confident = avg_score < SIMILARITY_THRESHOLD

    print(
        f"\n[evaluation] avg={avg_score:.4f} | threshold={SIMILARITY_THRESHOLD} "
        f"| confident={is_confident}"
    )

    return {
        "is_confident": is_confident,
        "hitl_needed" : not is_confident,
    }


# ── Node 4: HITL (stub — real interaction happens in Streamlit UI) ────────────

def hitl_node(state: RAGState) -> dict:
    """
    In CLI mode this is a no-op stub.
    In Streamlit, the UI injects hitl_answer directly into the state
    before displaying to the user — no blocking input() needed.
    """
    print("[hitl] Low-confidence answer flagged for human review.")
    return {}
