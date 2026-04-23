"""
groq_client.py — Groq LLM interface for RAG answer generation.

Uses llama-3.3-70b-versatile for fast, accurate inference.
Strictly grounded in context — hallucination prevention built into prompt.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL
from llm.prompt_template import build_prompt

_client = Groq(api_key=GROQ_API_KEY)


def generate_answer(query: str, context: list[str], instruction: str = "") -> str:
    """
    Generate a grounded answer using Groq LLM.
    
    Args:
        query: User's question
        context: List of retrieved document chunks
        instruction: Optional human instruction (for HITL re-generation)
    
    Returns:
        Generated answer string
    """
    if not context:
        return "I could not find relevant information in the provided documents."

    prompt = build_prompt(query, context, instruction)

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
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[groq_client] Error: {e}")
        return f"LLM Error: {str(e)}"
