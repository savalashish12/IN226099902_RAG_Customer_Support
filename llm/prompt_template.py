"""
prompt_template.py — Builds the grounded answer prompt.

Design:
  - Allows partial answers when context is incomplete
  - Clean numbered chunk format
  - Hard fallback only if truly no information exists
"""

from typing import List

SYSTEM_INSTRUCTION = (
    "You are a customer support assistant.\n\n"
    "Answer using ONLY the provided context.\n"
    "If partial information exists, explain clearly based on what is available.\n"
    "If the answer is truly not present in the context, say exactly:\n"
    "\"I don't have enough information.\""
)


def build_prompt(query: str, context: List[str]) -> str:
    if not context:
        context_block = "[No context retrieved]"
    else:
        # Number each chunk and limit length to keep prompt tight
        context_block = "\n".join(
            f"[{i+1}] {chunk[:600].strip()}"
            for i, chunk in enumerate(context)
        )

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Answer:"
    )
