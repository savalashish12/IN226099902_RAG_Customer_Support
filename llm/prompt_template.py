"""
prompt_template.py — Builds the grounded answer prompt.

Design:
  - Allows partial answers when context is incomplete
  - Clean numbered chunk format
  - Hard fallback only if truly no information exists
"""

from typing import List

def build_prompt(query: str, context: List[str], instruction: str = "") -> str:
    system_instruction = (
        "Answer ONLY from the provided context.\n"
        "If the answer is not present, say:\n"
        "'I could not find this in the provided documents.'\n"
        "Do NOT use external knowledge."
    )
    
    if not context:
        context_block = "[No context retrieved]"
    else:
        # Number each chunk and limit length to keep prompt tight
        context_block = "\n".join(
            f"[{i+1}] {chunk[:600].strip()}"
            for i, chunk in enumerate(context)
        )

    prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query.strip()}\n"
    )
    
    if instruction:
        prompt += f"\nInstruction:\n{instruction.strip()}\n"
        
    prompt += "\nAnswer:"
    return prompt
