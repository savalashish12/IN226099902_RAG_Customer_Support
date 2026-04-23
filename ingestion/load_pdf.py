"""
load_pdf.py — Loads a PDF from bytes, chunks it, embeds it, stores in ChromaDB.
Uses MD5 hash for deduplication — same file never re-embedded.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from langchain_community.document_loaders import PyPDFLoader
from ingestion.chunking import split_documents
from ingestion.vector_store import store_chunks, collection_exists


def load_and_ingest_pdf(file_bytes: bytes, collection_name: str, filename: str = "document.pdf") -> int:
    """
    Load PDF bytes → chunk → embed → store in ChromaDB.
    
    Args:
        file_bytes: Raw PDF bytes from Streamlit uploader
        collection_name: Target ChromaDB collection (doc_<md5>)
        filename: Original filename for metadata
    
    Returns:
        Number of chunks stored
    """
    # Save to temp file (PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader    = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Add source metadata
        for doc in documents:
            doc.metadata["source"]   = filename
            doc.metadata["filename"] = filename

        chunks = split_documents(documents)
        count  = store_chunks(chunks, collection_name)
        return count

    finally:
        os.unlink(tmp_path)
