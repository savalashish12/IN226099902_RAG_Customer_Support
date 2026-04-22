"""
config.py — Central configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Provider: "gemini" or "groq" ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

# --- Gemini ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Groq ---
GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings (always Gemini) ---
EMBEDDING_MODEL: str    = "gemini-embedding-001"
EMBEDDING_BATCH_SIZE    = int(os.getenv("EMBEDDING_BATCH_SIZE", "25"))
EMBEDDING_DELAY_SECONDS = int(os.getenv("EMBEDDING_DELAY_SECONDS", "5"))

# --- ChromaDB ---
CHROMA_DB_PATH: str   = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME: str  = "rag_default"

# --- Chunking ---
CHUNK_SIZE: int    = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_CHUNKS: int    = int(os.getenv("MAX_CHUNKS", "60"))

# --- Retrieval ---
TOP_K: int                    = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD: float   = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
