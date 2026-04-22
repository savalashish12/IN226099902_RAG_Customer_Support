"""
load_pdf.py — Loads PDF files.
Accepts a specific file path (Streamlit upload) or falls back to DATA_DIR scan.
"""

import os
import sys
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = "./data"


def load_pdfs(file_path: str = None) -> List[Document]:
    """
    Load a single PDF by path, or all PDFs in DATA_DIR.
    Returns list of LangChain Document objects (one per page).
    """
    documents = []

    if file_path:
        print(f"  [load_pdf] Loading: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        pages  = loader.load()
        documents.extend(pages)
        print(f"  [load_pdf] Pages loaded: {len(pages)}")
    else:
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in '{DATA_DIR}'")

        for pdf_file in pdf_files:
            path   = os.path.join(DATA_DIR, pdf_file)
            loader = PyPDFLoader(path)
            pages  = loader.load()
            documents.extend(pages)
            print(f"  [load_pdf] {pdf_file} → {len(pages)} pages")

    print(f"  [load_pdf] Total pages: {len(documents)}")
    return documents
