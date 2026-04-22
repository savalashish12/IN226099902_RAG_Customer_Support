"""
state.py — Shared state flowing through the LangGraph pipeline.
"""

from typing import List
from typing_extensions import TypedDict


class RAGState(TypedDict):
    query          : str
    collection_name: str
    context        : List[str]       # retrieved chunk texts
    scores         : List[float]     # L2 distances (lower = more relevant)
    answer         : str
    is_confident   : bool
    hitl_needed    : bool
    hitl_answer    : str             # human override (empty if not used)
