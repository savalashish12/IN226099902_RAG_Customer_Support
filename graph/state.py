"""
state.py — LangGraph shared state for RAG pipeline.
All nodes read from and write to this TypedDict.
"""
from typing import List, Optional
from typing_extensions import TypedDict


class RAGState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    query            : str              # User's question
    collection_name  : str              # ChromaDB collection (doc_<md5>)

    # ── Retrieval Output ──────────────────────────────────────
    context          : List[str]        # Retrieved chunk texts
    scores           : List[float]      # L2 distances (lower = more relevant)
    sources          : List[dict]       # Chunk metadata (page, source)

    # ── Generation Output ─────────────────────────────────────
    answer           : str              # LLM-generated answer

    # ── Evaluation Output ─────────────────────────────────────
    avg_score        : float            # Mean of scores list
    similarity_pct   : float            # (1 - avg_score) * 100, capped 0-100
    confidence_level : str              # "HIGH" | "MEDIUM" | "LOW"
    is_confident     : bool             # True if HIGH or MEDIUM (no HITL)
    hitl_needed      : bool             # True if HITL should trigger
    hitl_reason      : str              # Why HITL was triggered

    # ── HITL Fields ───────────────────────────────────────────
    hitl_instruction : str              # Human instruction for re-generation
    hitl_answer      : str              # Human-approved final answer
    thread_id        : str              # LangGraph checkpoint thread ID
