"""
config.py — Central configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

# LLM
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ChromaDB
CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Chunking
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_CHUNKS      = int(os.getenv("MAX_CHUNKS", 60))

# Retrieval
TOP_K           = int(os.getenv("TOP_K", 5))

# Confidence Thresholds (L2 Distance — lower = better)
HIGH_CONF_THRESHOLD   = float(os.getenv("HIGH_CONF_THRESHOLD",   0.30))
MEDIUM_CONF_THRESHOLD = float(os.getenv("MEDIUM_CONF_THRESHOLD", 0.55))

# Embedding
EMBEDDING_MODEL        = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_BATCH_SIZE   = int(os.getenv("EMBEDDING_BATCH_SIZE",   25))
EMBEDDING_DELAY_SECONDS= float(os.getenv("EMBEDDING_DELAY_SECONDS", 5))
