"""
batch.py — Batch ingestion for multi-file, folder-path, and ZIP uploads.

Supports:
  - Multiple PDF files at once (Streamlit multi-upload)
  - Local folder scanning (os.walk)
  - ZIP archive extraction (PDF members only)

Each file is MD5-deduplicated independently — the same document is
never re-embedded, across batches and sessions.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
import io
import zipfile
from dataclasses import dataclass

from ingestion.load_pdf import load_and_ingest_pdf
from ingestion.vector_store import collection_exists


@dataclass
class IngestResult:
    """Outcome of ingesting a single PDF in a batch."""
    name       : str
    md5        : str
    status     : str              # "ingested" | "cached" | "failed"
    chunks     : int = 0
    collection : str = ""
    error      : str = ""


def get_file_md5(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:16]


def _is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")


def discover_pdfs_in_folder(folder_path: str) -> list[tuple[str, bytes]]:
    """
    Recursively collect (filename, bytes) for every PDF under folder_path.
    Raises ValueError if the folder does not exist.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    items: list[tuple[str, bytes]] = []
    for root, _dirs, files in os.walk(folder_path):
        for fname in sorted(files):
            if not _is_pdf(fname):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, "rb") as f:
                items.append((fname, f.read()))
    return items


def extract_pdfs_from_zip(zip_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Extract every PDF member from a ZIP archive as (filename, bytes).
    Skips directories, non-PDF entries, and macOS junk (__MACOSX / ._*).
    Raises ValueError for invalid/corrupted ZIP data.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid or corrupted ZIP file.") from e

    items: list[tuple[str, bytes]] = []
    with zf:
        for info in zf.infolist():
            if info.is_dir() or not _is_pdf(info.filename):
                continue
            parts = info.filename.replace("\\", "/").split("/")
            base = parts[-1]
            if "__MACOSX" in parts or base.startswith("._"):
                continue
            items.append((base, zf.read(info)))
    return items


def ingest_one(file_bytes: bytes, filename: str) -> IngestResult:
    """MD5-dedup check, then chunk → embed → store. Never re-embeds."""
    md5         = get_file_md5(file_bytes)
    collection  = f"doc_{md5}"

    if collection_exists(collection):
        return IngestResult(name=filename, md5=md5, status="cached",
                            collection=collection)

    try:
        chunks = load_and_ingest_pdf(file_bytes, collection, filename)
        return IngestResult(name=filename, md5=md5, status="ingested",
                            chunks=chunks, collection=collection)
    except Exception as e:
        return IngestResult(name=filename, md5=md5, status="failed",
                            error=str(e))


def ingest_batch(items: list[tuple[str, bytes]],
                 progress_callback=None) -> list[IngestResult]:
    """
    Ingest a batch of (filename, bytes) PDFs sequentially.
    progress_callback(i, total, filename) fires before each file.
    """
    results: list[IngestResult] = []
    total = len(items)
    for i, (name, data) in enumerate(items):
        if progress_callback:
            progress_callback(i, total, name)
        results.append(ingest_one(data, name))
    return results
