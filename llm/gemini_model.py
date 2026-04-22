"""
gemini_model.py — LLM generation using Gemini or Groq (configurable via LLM_PROVIDER).
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenacity import retry, wait_exponential, stop_after_attempt
from app.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY,   GROQ_MODEL,
)
from llm.prompt_template import build_prompt


def generate_answer(query: str, context: List[str]) -> str:
    """
    Generate a grounded answer using the configured LLM provider.
    Supports: 'gemini' or 'groq'
    """
    prompt = build_prompt(query, context)

    if LLM_PROVIDER == "groq":
        return _call_groq(prompt)
    else:
        return _call_gemini(prompt)


# ── Groq ──────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY missing — check .env")

    client = Groq(api_key=GROQ_API_KEY)

    @retry(wait=wait_exponential(multiplier=2, min=4, max=30), stop=stop_after_attempt(4), reraise=True)
    def _call():
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    answer = _call()
    print(f"  [llm/groq] Answer generated ({len(answer)} chars) via {GROQ_MODEL}")
    return answer


# ── Gemini ────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing — check .env")

    client = genai.Client(api_key=GEMINI_API_KEY)

    @retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5), reraise=True)
    def _call():
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
        )
        return response.text.strip()

    answer = _call()
    print(f"  [llm/gemini] Answer generated ({len(answer)} chars) via {GEMINI_MODEL}")
    return answer
