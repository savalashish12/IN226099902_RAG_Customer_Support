"""
nodes.py — Four LangGraph nodes: retrieval → generation → evaluation → hitl

IMPORTANT HITL DESIGN:
  - Uses LangGraph's interrupt() to PAUSE the graph
  - Graph resumes via Command(resume={...}) from Streamlit UI
  - MemorySaver checkpointer saves state between pause and resume
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.types import interrupt
from app.config import HIGH_CONF_THRESHOLD, MEDIUM_CONF_THRESHOLD
from retrieval.retriever import retrieve_context_with_scores
from llm.groq_client import generate_answer
from graph.state import RAGState


# ═══════════════════════════════════════════════════════════════
# NODE 1: RETRIEVAL
# ═══════════════════════════════════════════════════════════════
def retrieval_node(state: RAGState) -> dict:
    """
    Fetches top-k semantically relevant chunks from ChromaDB.
    Uses MMR (Maximal Marginal Relevance) for diversity.
    Returns chunks with L2 distance scores (lower = more relevant).
    """
    query           = state["query"]
    collection_name = state["collection_name"]

    print(f"\n[retrieval] Query: '{query}' | Collection: {collection_name}")

    results = retrieve_context_with_scores(query, collection_name=collection_name)

    context = [text     for text, score, meta in results]
    scores  = [score    for text, score, meta in results]
    sources = [meta     for text, score, meta in results]

    print(f"[retrieval] {len(context)} chunks | Scores: {[round(s,3) for s in scores]}")

    return {"context": context, "scores": scores, "sources": sources}


# ═══════════════════════════════════════════════════════════════
# NODE 2: GENERATION
# ═══════════════════════════════════════════════════════════════
def generation_node(state: RAGState) -> dict:
    """
    Calls Groq LLM with retrieved context to generate an answer.
    
    NO-ANSWER GUARD: If avg score is very high (> MEDIUM threshold),
    the chunks are too dissimilar — skip LLM and return a standard
    "not found" response. This prevents hallucination.
    """
    query       = state["query"]
    context     = state["context"]
    scores      = state.get("scores", [])
    instruction = state.get("hitl_instruction", "")

    avg_score = sum(scores) / len(scores) if scores else 1.0

    # No-Answer Guard — chunks too dissimilar, don't call LLM
    if avg_score >= MEDIUM_CONF_THRESHOLD:
        print(f"[generation] No-Answer Guard | avg_score={avg_score:.4f} >= {MEDIUM_CONF_THRESHOLD}")
        return {
            "answer": "I could not find relevant information in the provided documents.",
            "avg_score": avg_score,
        }

    print(f"[generation] avg_score={avg_score:.4f} — calling LLM...")
    answer = generate_answer(query, context, instruction)
    print(f"[generation] Answer (preview): {answer[:120]}...")

    return {"answer": answer, "avg_score": avg_score}


# ═══════════════════════════════════════════════════════════════
# NODE 3: EVALUATION
# ═══════════════════════════════════════════════════════════════
def _classify_confidence(avg_score: float, answer: str) -> tuple[str, bool, bool, str]:
    """
    Maps avg L2 distance → confidence label, HITL decision.
    
    Returns: (confidence_level, is_confident, hitl_needed, hitl_reason)
    
    L2 Distance Logic (ChromaDB default metric):
      0.00 – 0.30 → HIGH   → Direct document match → No HITL
      0.30 – 0.55 → MEDIUM → Partial match         → HITL only if complex
      0.55+       → LOW    → No match              → Always HITL
    """
    not_found_phrases = [
        "could not find", "not in the provided", "don't know",
        "no information", "not mentioned"
    ]
    is_no_answer = any(p in answer.lower() for p in not_found_phrases)

    if is_no_answer or avg_score >= MEDIUM_CONF_THRESHOLD:
        return "LOW", False, True, "Query out of document scope"

    if avg_score < HIGH_CONF_THRESHOLD:
        return "HIGH", True, False, "Direct document match"

    # MEDIUM — partial match
    return "MEDIUM", True, False, "Partial document match"


def evaluation_node(state: RAGState) -> dict:
    """
    Evaluates retrieval quality and decides whether to trigger HITL.
    
    HITL Escalation Criteria:
      1. LOW confidence  → always escalate
      2. MEDIUM + query has urgent/error keywords → escalate
      3. MEDIUM + query > 10 words (complex)  → escalate
      4. Answer contains uncertainty phrases  → escalate
    """
    scores  = state.get("scores", [])
    query   = state.get("query", "")
    answer  = state.get("answer", "")

    avg_score      = state.get("avg_score") or (sum(scores) / len(scores) if scores else 1.0)
    similarity_pct = round(max(0.0, (1.0 - avg_score) * 100), 1)

    confidence_level, is_confident, hitl_needed, hitl_reason = \
        _classify_confidence(avg_score, answer)

    # Additional HITL escalation rules for MEDIUM confidence
    if confidence_level == "MEDIUM":
        escalation_keywords = [
            "urgent", "critical", "emergency", "not working", "failed",
            "error", "broken", "crash", "data loss", "security breach"
        ]
        if any(kw in query.lower() for kw in escalation_keywords):
            hitl_needed = True
            hitl_reason = "Urgent/critical keyword detected"
        elif len(query.split()) > 10:
            hitl_needed = True
            hitl_reason = "Complex multi-part query"

    print(
        f"\n[evaluation] avg={avg_score:.4f} | similarity={similarity_pct}% "
        f"| confidence={confidence_level} | hitl={hitl_needed} | reason={hitl_reason}"
    )

    return {
        "avg_score"       : avg_score,
        "similarity_pct"  : similarity_pct,
        "confidence_level": confidence_level,
        "is_confident"    : is_confident,
        "hitl_needed"     : hitl_needed,
        "hitl_reason"     : hitl_reason,
    }


# ═══════════════════════════════════════════════════════════════
# NODE 4: HITL (Human-in-the-Loop)
# ═══════════════════════════════════════════════════════════════
def hitl_node(state: RAGState) -> dict:
    """
    REAL LangGraph HITL using interrupt().
    
    HOW IT WORKS:
      1. Graph execution PAUSES here via interrupt()
      2. The interrupt payload is sent to the Streamlit UI
      3. Streamlit shows the HITL panel to the human agent
      4. Human reviews and submits → UI calls graph.invoke(Command(resume={...}))
      5. Graph RESUMES from this exact point with the human's input
      6. Returns updated answer to the user
    
    The MemorySaver checkpointer saves the full graph state between
    pause and resume, identified by thread_id in the config.
    """
    print("[hitl] ⏸️  Graph paused — waiting for human review...")

    # PAUSE EXECUTION — sends payload to Streamlit UI
    human_input = interrupt({
        "query"       : state["query"],
        "ai_answer"   : state["answer"],
        "confidence"  : state["confidence_level"],
        "similarity"  : state.get("similarity_pct", 0),
        "reason"      : state.get("hitl_reason", ""),
        "chunks_used" : len(state.get("context", [])),
    })

    # RESUMES HERE after Command(resume={...}) is called
    print(f"[hitl] ▶️  Resumed with human input: {str(human_input)[:80]}...")

    # Human can: approve AI answer, provide corrected answer, or add instructions
    approved_answer = human_input.get("approved_answer", state["answer"])
    was_overridden  = human_input.get("overridden", False)

    return {
        "answer"      : approved_answer,
        "hitl_answer" : approved_answer,
        "hitl_needed" : True,
        "hitl_reason" : state.get("hitl_reason", "") + (" [Human Overridden]" if was_overridden else " [Human Approved]"),
    }
