"""
retriever.py — Retrieves top-k chunks with scores and metadata from a named ChromaDB collection.
"""

import os
import sys
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_chroma import Chroma
from app.config import CHROMA_DB_PATH, TOP_K
from ingestion.embedding import get_embedding_function

# Cache: collection_name → Chroma instance
_cache: Dict[str, Chroma] = {}


def _get_vectorstore(collection_name: str) -> Chroma:
    if collection_name not in _cache:
        embedding_fn = get_embedding_function()
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=CHROMA_DB_PATH,
        )
        count = vs._collection.count()
        print(f"  [retriever] Loaded '{collection_name}' — {count} chunks")
        _cache[collection_name] = vs
    return _cache[collection_name]


def retrieve_context_with_scores(
    query: str,
    collection_name: str,
    top_k: int = TOP_K,
) -> List[Tuple[str, float, dict]]:
    """
    Returns list of (text, score, metadata) tuples.
    Uses MMR search for diversity. Converts L2 distance to Similarity Score.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    vs = _get_vectorstore(collection_name)
    
    # 1. Use MMR search to ensure diversity via as_retriever
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 15
        }
    )
    mmr_docs = retriever.invoke(query)
    
    # 2. Fetch scores using standard search to map to MMR docs
    # Chroma returns L2 distances by default
    all_results = vs.similarity_search_with_score(query, k=15)
    score_map = {}
    for doc, distance in all_results:
        # Use raw L2 distance (lower is better)
        score_map[doc.page_content] = distance

    output = []
    for doc in mmr_docs:
        sim = score_map.get(doc.page_content, 0.0)
        output.append((doc.page_content, round(sim, 4), doc.metadata))

    return output
