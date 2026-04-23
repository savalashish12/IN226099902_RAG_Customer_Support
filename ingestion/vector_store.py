"""
vector_store.py — ChromaDB CRUD operations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_chroma import Chroma
from app.config import CHROMA_DB_PATH
from ingestion.embedding import get_embedding_function


def _get_store(collection_name: str) -> Chroma:
    return Chroma(
        collection_name   = collection_name,
        embedding_function= get_embedding_function(),
        persist_directory = CHROMA_DB_PATH,
    )


def store_chunks(chunks: list, collection_name: str) -> int:
    """Embed and store chunks. Returns number stored."""
    store = _get_store(collection_name)
    texts = [c.page_content for c in chunks]
    metas = [c.metadata     for c in chunks]
    ids   = [f"{collection_name}_chunk_{i}" for i in range(len(texts))]
    store.add_texts(texts=texts, metadatas=metas, ids=ids)
    return len(texts)


def collection_exists(collection_name: str) -> bool:
    """Check if a collection already has chunks (dedup guard)."""
    try:
        store = _get_store(collection_name)
        return store._collection.count() > 0
    except Exception:
        return False


def list_collections() -> list[str]:
    """List all persisted ChromaDB collections."""
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return [c.name for c in client.list_collections()]


def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
