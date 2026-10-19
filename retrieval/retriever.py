"""
retriever.py — Retrieves top-k chunks with scores and metadata from a named ChromaDB collection.
"""

import os
import sys
from typing import Dict, List, Tuple

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


def retrieve_multi_context_with_scores(
    query: str,
    collection_names: List[str],
    top_k: int = TOP_K,
) -> List[Tuple[str, float, dict]]:
    """
    Search across MULTIPLE collections and merge results.

    Each collection is searched via MMR; results from all collections are
    combined, sorted by L2 distance (lower = more relevant), and the
    globally best top_k chunks are returned. Metadata is enriched with the
    originating collection name so the answer can cite its source.

    A single-collection input transparently falls back to standard retrieval.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not collection_names:
        return []

    if len(collection_names) == 1:
        return retrieve_context_with_scores(query, collection_names[0], top_k)

    merged: List[Tuple[str, float, dict]] = []
    for name in collection_names:
        try:
            results = retrieve_context_with_scores(query, name, top_k)
        except Exception:
            continue
        for text, score, meta in results:
            meta = dict(meta or {})
            meta.setdefault("collection", name)
            merged.append((text, score, meta))

    # Deduplicate identical chunk text (same doc may appear in multiple collections)
    seen = set()
    deduped = []
    for text, score, meta in merged:
        if text in seen:
            continue
        seen.add(text)
        deduped.append((text, score, meta))

    deduped.sort(key=lambda x: x[1])
    return deduped[:top_k]
