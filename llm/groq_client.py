"""
groq_client.py — Groq LLM interface for RAG answer generation.

Uses llama-3.3-70b-versatile for fast, accurate inference.
Strictly grounded in context — hallucination prevention built into prompt.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from llm.prompt_template import build_prompt

_client = Groq(api_key=GROQ_API_KEY)


def generate_answer(query: str, context: list[str], instruction: str = "",
                    sources: list[dict] | None = None) -> str:
    """
    Generate a grounded answer using Groq LLM.

    Args:
        query: User's question
        context: List of retrieved document chunks
        instruction: Optional human instruction (for HITL re-generation)
        sources: Optional per-chunk metadata (filename/page) for citations

    Returns:
        Generated answer string
    """
    if not context:
        return "I could not find relevant information in the provided documents."

    prompt = build_prompt(query, context, instruction, sources)

    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful enterprise support assistant for TechNova Solutions. "
                        "Answer ONLY from the provided context. "
                        "If the answer is not in the context, say exactly: "
                        "'I could not find relevant information in the provided documents.' "
                        "Never make up information. Be concise and professional."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,     # Low temperature for factual accuracy
            max_tokens=2048,     # Headroom for reasoning-style models (gpt-oss)
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    except Exception as e:
        print(f"[groq_client] Error: {e}")
        return f"LLM Error: {str(e)}"


def stream_answer(query: str, context: list[str], instruction: str = "",
                  sources: list[dict] | None = None):
    """
    Stream a grounded answer token-by-token from Groq.

    Yields string chunks as they arrive. Intended for the UI layer
    (Streamlit's st.write_stream) to render answers progressively.

    Args:
        query: User's question
        context: Retrieved document chunks
        instruction: Optional human instruction
        sources: Optional per-chunk metadata for citations

    Yields:
        Partial answer strings.
    """
    if not context:
        yield "I could not find relevant information in the provided documents."
        return

    prompt = build_prompt(query, context, instruction, sources)

    try:
        stream = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful enterprise support assistant for TechNova Solutions. "
                        "Answer ONLY from the provided context. "
                        "If the answer is not in the context, say exactly: "
                        "'I could not find relevant information in the provided documents.' "
                        "Never make up information. Be concise and professional."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        yield f"LLM Error: {str(e)}"
