"""
vector_store.py — Persistent ChromaDB with per-session collections.
"""

import os
import sys
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.config import CHROMA_DB_PATH


def store_in_chromadb(
    chunks: List[Document],
    embedding_fn,
    collection_name: str,
) -> Chroma:
    """Embed and store chunks in a named ChromaDB collection."""
    print(f"  [vector_store] Storing {len(chunks)} chunks → collection='{collection_name}'")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"  [vector_store] ✅ Done — {len(chunks)} chunks stored.")
    return vectorstore


def collection_exists(collection_name: str) -> bool:
    """Check if a named collection already exists in ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        existing = [c.name for c in client.list_collections()]
        return collection_name in existing
    except Exception:
        return False


def delete_collection(collection_name: str) -> None:
    """Delete a named collection from ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(collection_name)
        print(f"  [vector_store] 🗑 Deleted collection: {collection_name}")
    except Exception as e:
        print(f"  [vector_store] Warning: Could not delete '{collection_name}': {e}")


def list_collections() -> List[str]:
    """Return names of all collections in the persistent ChromaDB."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return [c.name for c in client.list_collections()]
    except Exception:
        return []


def load_vectorstore(embedding_fn, collection_name: str) -> Chroma:
    """Load an existing collection for retrieval."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DB_PATH,
    )
