"""
embedding.py — Gemini embedding with safe batching, delay, and exponential retry.
"""

import os
import sys
import time
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DELAY_SECONDS

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


class RateLimitedEmbeddings(Embeddings):
    """Gemini embeddings wrapper with batching, delay, and retry."""

    def __init__(self, base: Embeddings, batch_size: int = 25, delay: int = 5):
        self.base       = base
        self.batch_size = batch_size
        self.delay      = delay

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        return self.base.embed_documents(batch)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results      = []
        total        = (len(texts) + self.batch_size - 1) // self.batch_size

        for idx in range(0, len(texts), self.batch_size):
            batch     = texts[idx : idx + self.batch_size]
            batch_num = idx // self.batch_size + 1
            print(f"  [embedding] Batch {batch_num}/{total} ({len(batch)} items)...")
            results.extend(self._embed_batch(batch))

            if idx + self.batch_size < len(texts):
                print(f"  [embedding] Waiting {self.delay}s...")
                time.sleep(self.delay)

        return results

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def embed_query(self, text: str) -> List[float]:
        return self.base.embed_query(text)


def get_embedding_function() -> Embeddings:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing — check .env")

    base = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
        task_type="retrieval_document",
    )

    print(f"  [embedding] Initialized (batch={EMBEDDING_BATCH_SIZE}, delay={EMBEDDING_DELAY_SECONDS}s)")
    return RateLimitedEmbeddings(base, EMBEDDING_BATCH_SIZE, EMBEDDING_DELAY_SECONDS)
