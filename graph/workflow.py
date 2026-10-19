"""
workflow.py — Builds LangGraph StateGraph with MemorySaver checkpointer.

KEY DESIGN:
  - MemorySaver enables interrupt/resume HITL pattern
  - thread_id in config identifies each conversation session
  - interrupt_before=["hitl"] pauses graph BEFORE entering hitl_node
  - Resume via graph.invoke(Command(resume={...}), config=config)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from graph.nodes import (
    evaluation_node,
    generation_node,
    hitl_node,
    retrieval_node,
    rewrite_node,
)
from graph.state import RAGState


# ── Router function ────────────────────────────────────────────────────────────
def _route_after_evaluation(state: RAGState) -> str:
    """Conditional edge: go to HITL if needed, else end."""
    if state.get("hitl_needed", False):
        return "hitl"
    return "end"


# ── Graph Definition ───────────────────────────────────────────────────────────
def build_graph():
    """
    Graph Flow:
      START → retrieval → generation → evaluation → (hitl → END) or END

    The graph compiles with MemorySaver so state is persisted between
    interrupt() pause and Command(resume=...) call.
    """
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("rewrite",    rewrite_node)
    builder.add_node("retrieval",  retrieval_node)
    builder.add_node("generation", generation_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("hitl",       hitl_node)

    # Add edges
    builder.add_edge(START,        "rewrite")
    builder.add_edge("rewrite",    "retrieval")
    builder.add_edge("retrieval",  "generation")
    builder.add_edge("generation", "evaluation")
    builder.add_edge("hitl",       END)

    # Conditional routing after evaluation
    builder.add_conditional_edges(
        "evaluation",
        _route_after_evaluation,
        {"end": END, "hitl": "hitl"},
    )

    # Compile with MemorySaver (enables interrupt/resume)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level graph instance
_graph = build_graph()


# ── Retrieval-only path (for UI-driven streaming) ─────────────────────────────
def run_retrieval_only(query: str, collection_name: str,
                       collection_names: list[str] | None = None,
                       chat_history: list[dict] | None = None) -> dict:
    """
    Run rewrite + retrieval only (no LLM generation, no HITL).

    Used by the UI streaming path: it retrieves context + scores, then the
    UI streams generation itself and evaluates HITL locally. Returns a dict
    with context/scores/sources, the standalone query, and provisional
    confidence derived from retrieval scores alone.

    Note: the no-answer guard (avg_score >= MEDIUM threshold) is applied here
    so obviously out-of-scope queries short-circuit to HITL without streaming.
    """
    from app.config import MEDIUM_CONF_THRESHOLD
    from graph.nodes import retrieval_node, rewrite_node

    if collection_names is None:
        collection_names = [collection_name] if collection_name else []

    base_state: RAGState = {
        "query": query, "collection_name": collection_name,
        "collection_names": collection_names,
        "chat_history": chat_history or [], "standalone_query": query,
    }
    base_state.update(rewrite_node(base_state))
    base_state.update(retrieval_node(base_state))

    scores = base_state.get("scores", [])
    avg_score = sum(scores) / len(scores) if scores else 1.0
    return {
        "standalone_query": base_state.get("standalone_query", query),
        "context"         : base_state.get("context", []),
        "scores"          : scores,
        "sources"         : base_state.get("sources", []),
        "avg_score"       : avg_score,
        "guard_no_answer" : avg_score >= MEDIUM_CONF_THRESHOLD,
    }


# ── Public API ─────────────────────────────────────────────────────────────────
def run_rag_pipeline(query: str, collection_name: str, thread_id: str,
                     collection_names: list[str] | None = None,
                     chat_history: list[dict] | None = None) -> dict:
    """
    Start a new RAG pipeline run.

    Args:
        query: User's question
        collection_name: Single ChromaDB collection ID (doc_<md5>)
        thread_id: Unique session ID for this conversation (enables checkpointing)
        collection_names: Optional list of collections to search across (unified KB).
        chat_history: Optional prior turns [{role, content}] for conversational memory.

    Returns:
        Final state dict. If hitl_needed=True, graph is PAUSED at hitl_node.
        Call resume_with_human_input() to continue.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if collection_names is None:
        collection_names = [collection_name] if collection_name else []

    initial_state: RAGState = {
        "query"           : query,
        "collection_name" : collection_name,
        "collection_names": collection_names,
        "chat_history"    : chat_history or [],
        "standalone_query": query,
        "context"         : [],
        "scores"          : [],
        "sources"         : [],
        "answer"          : "",
        "avg_score"       : 0.0,
        "similarity_pct"  : 0.0,
        "confidence_level": "LOW",
        "is_confident"    : False,
        "hitl_needed"     : False,
        "hitl_answer"     : "",
        "hitl_instruction": "",
        "hitl_reason"     : "",
        "thread_id"       : thread_id,
    }

    result = _graph.invoke(initial_state, config=config)
    return result


def resume_with_human_input(thread_id: str, approved_answer: str, overridden: bool = False) -> dict:
    """
    Resume a PAUSED graph after human review.

    This is called by the Streamlit UI when the human agent clicks
    "Approve" or "Override" in the HITL panel.

    Args:
        thread_id: Same thread_id used in run_rag_pipeline()
        approved_answer: The final answer (human's or AI's)
        overridden: True if human wrote a new answer, False if approved AI answer

    Returns:
        Final state after resumption
    """
    config = {"configurable": {"thread_id": thread_id}}

    result = _graph.invoke(
        Command(resume={
            "approved_answer": approved_answer,
            "overridden"     : overridden,
        }),
        config=config,
    )
    return result


def get_graph_state(thread_id: str) -> dict:
    """Get current state of a paused graph (for debugging/inspection)."""
    config = {"configurable": {"thread_id": thread_id}}
    state = _graph.get_state(config)
    return state.values if state else {}
