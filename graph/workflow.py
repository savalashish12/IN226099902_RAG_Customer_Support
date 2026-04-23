"""
workflow.py — Builds LangGraph StateGraph with MemorySaver checkpointer.

KEY DESIGN:
  - MemorySaver enables interrupt/resume HITL pattern
  - thread_id in config identifies each conversation session
  - interrupt_before=["hitl"] pauses graph BEFORE entering hitl_node
  - Resume via graph.invoke(Command(resume={...}), config=config)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from graph.state import RAGState
from graph.nodes import retrieval_node, generation_node, evaluation_node, hitl_node

# ── Router function ────────────────────────────────────────────────────────────
def _route_after_evaluation(state: RAGState) -> str:
    """Conditional edge: go to HITL if needed, else end."""
    if state.get("hitl_needed", False):
        print("[router] → hitl_node")
        return "hitl"
    print("[router] → END (confident)")
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
    builder.add_node("retrieval",  retrieval_node)
    builder.add_node("generation", generation_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("hitl",       hitl_node)

    # Add edges
    builder.add_edge(START,        "retrieval")
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


# ── Public API ─────────────────────────────────────────────────────────────────
def run_rag_pipeline(query: str, collection_name: str, thread_id: str) -> dict:
    """
    Start a new RAG pipeline run.
    
    Args:
        query: User's question
        collection_name: ChromaDB collection ID (doc_<md5>)
        thread_id: Unique session ID for this conversation (enables checkpointing)
    
    Returns:
        Final state dict. If hitl_needed=True, graph is PAUSED at hitl_node.
        Call resume_with_human_input() to continue.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: RAGState = {
        "query"           : query,
        "collection_name" : collection_name,
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
