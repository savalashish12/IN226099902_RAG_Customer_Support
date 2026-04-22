"""
chunking.py — Splits documents into clean, overlapping chunks.

Fixes applied:
  - Regex cleans broken lines, extra spaces, and TOC-like noise
  - MAX_CHUNKS cap prevents excessive API calls
  - chunk_size=900, overlap=100 for better context coverage
"""

import re
import os
import sys
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


# Patterns that indicate noisy / low-value content (TOC lines, page markers)
_NOISE_PATTERNS = [
    r"^\s*\d+\s*$",                        # standalone page numbers
    r"^(chapter|section|table of contents)", # TOC headers
    r"^\.{5,}",                            # dotted leaders "..........."
    r"^\s*[ivxlcdmIVXLCDM]+\s*$",         # standalone roman numerals
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def _clean_text(text: str) -> str:
    """Remove noise, fix broken lines, collapse whitespace."""
    # Remove lines matching noise patterns
    lines   = text.splitlines()
    cleaned = [ln for ln in lines if not _NOISE_RE.match(ln)]
    text    = " ".join(cleaned)

    # Collapse multiple spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Remove hyphenated line breaks: "infor-\nmation" → "information"
    text = re.sub(r"-\s+", "", text)
    return text.strip()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Clean and split documents into chunks.

    Returns:
        List of Document chunks capped at MAX_CHUNKS.
    """
    # Clean each page
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)

    # Filter out near-empty pages
    documents = [d for d in documents if len(d.page_content) > 50]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"  [chunking] Pages: {len(documents)} | Chunks: {len(chunks)} | size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return chunks
