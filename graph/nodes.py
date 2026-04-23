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
    for i, (c, s) in enumerate(zip(context, scores)):
        print(f"  [{i+1}] score={s:.4f} | {c[:100].replace(chr(10), ' ')}...")

    return {"context": context, "scores": scores}


# ── Node 2: Generation ────────────────────────────────────────────────────────

def generation_node(state: RAGState) -> dict:
    query   = state["query"]
    context = state["context"]
    instruction = state.get("hitl_instruction", "")
    scores  = state.get("scores", [])

    print(f"\n[generation] Generating answer...")
    
    avg_score = sum(scores) / len(scores) if scores else 1.0
    if avg_score > 0.55:
        print(f"[generation] No-Answer Guard triggered! avg_score={avg_score:.4f} > 0.55")
        return {
            "answer": "I could not find this in the provided documents.",
            "confidence_level": "LOW",
            "hitl_needed": True
        }
    
    answer = generate_answer(query, context, instruction)
        
    print(f"[generation] Answer: {answer[:120]}...")

    return {"answer": answer}


def should_trigger_hitl(query: str, answer: str, avg_score: float, confidence: str):
    if confidence == "LOW":
        return True, "Low confidence answer"
        
    if confidence == "MEDIUM" and len(query.split()) > 8:
        return True, "Complex query needs clarification"
        
    if any(x in answer.lower() for x in [
        "could not find", "don't know", "not enough information"
    ]):
        return True, "AI uncertainty"
        
    keywords = ["urgent", "issue", "problem", "error", "failed", "not working"]
    if any(word in query.lower() for word in keywords):
        return True, "User escalation detected"
        
    return False, "Confident response"


# ── Node 3: Evaluation ────────────────────────────────────────────────────────

def evaluation_node(state: RAGState) -> dict:
    scores = state.get("scores", [])
    query = state.get("query", "")
    answer = state.get("answer", "")

    # Default to 1.0 (worst distance) if no scores
    avg_score = sum(scores) / len(scores) if scores else 1.0

    # 1. Confidence Mapping — lower L2 distance = better match
    if avg_score < 0.35:
        confidence_level = "HIGH"
    elif avg_score < 0.55:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # 2. Override: no-answer response always → LOW
    if "I could not find this in the provided documents." in answer:
        confidence_level = "LOW"

    # 3. Hybrid HITL Logic
    hitl_needed, hitl_reason = should_trigger_hitl(query, answer, avg_score, confidence_level)

    is_confident = confidence_level in ("HIGH", "MEDIUM")

    print(
        f"\n[evaluation] avg={avg_score:.4f} | confidence={confidence_level} "
        f"| hitl={hitl_needed} | reason={hitl_reason}"
    )

    return {
        "is_confident": is_confident,
        "hitl_needed" : hitl_needed,
        "confidence_level": confidence_level,
        "hitl_reason": hitl_reason
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
