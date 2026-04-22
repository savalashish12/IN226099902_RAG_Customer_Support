"""
workflow.py — Builds and runs the LangGraph RAG pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, START, END
from graph.state import RAGState
from graph.nodes import retrieval_node, generation_node, evaluation_node, hitl_node


def _router(state: RAGState) -> str:
    if state["is_confident"]:
        print("[router] Confident → END")
        return "end"
    print("[router] Low confidence → HITL")
    return "hitl"


def build_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieval",  retrieval_node)
    g.add_node("generation", generation_node)
    g.add_node("evaluation", evaluation_node)
    g.add_node("hitl",       hitl_node)

    g.add_edge(START,        "retrieval")
    g.add_edge("retrieval",  "generation")
    g.add_edge("generation", "evaluation")

    g.add_conditional_edges(
        "evaluation",
        _router,
        {"end": END, "hitl": "hitl"},
    )
    g.add_edge("hitl", END)

    return g.compile()


_graph = build_graph()


def run_rag_pipeline(query: str, collection_name: str) -> dict:
    """Run the full RAG pipeline and return final state."""
    initial: RAGState = {
        "query"          : query,
        "collection_name": collection_name,
        "context"        : [],
        "scores"         : [],
        "answer"         : "",
        "is_confident"   : False,
        "hitl_needed"    : False,
        "hitl_answer"    : "",
    }
    return _graph.invoke(initial)
