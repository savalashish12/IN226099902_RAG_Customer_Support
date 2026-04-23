"""
prompt_template.py — Builds structured prompts for the LLM.
"""


def build_prompt(query: str, context: list[str], instruction: str = "") -> str:
    """
    Build a RAG prompt that strictly grounds the LLM in retrieved context.
    
    The prompt uses XML-style tags to clearly delimit context and question,
    which improves LLM accuracy and reduces hallucination.
    """
    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context)
    )

    instruction_block = ""
    if instruction:
        instruction_block = f"\n\n<human_instruction>\n{instruction}\n</human_instruction>"

    return f"""You are answering a question for an enterprise support system.
Use ONLY the information in the context below to answer.
Do NOT use any external knowledge.

<context>
{context_text}
</context>{instruction_block}

<question>
{query}
</question>

Provide a clear, structured answer. If the context contains step-by-step instructions, present them as numbered steps. Be professional and concise."""
