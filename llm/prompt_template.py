"""
prompt_template.py — Builds structured prompts for the LLM.
"""


def _source_label(meta: dict | None) -> str:
    """Build a human-readable source label: 'file.pdf, p.4'."""
    if not meta:
        return "unknown"
    name = meta.get("source") or meta.get("filename") or meta.get("collection") or "doc"
    page = meta.get("page")
    if isinstance(page, int):
        return f"{name}, p.{page + 1}"
    return str(name)


def build_prompt(query: str, context: list[str], instruction: str = "",
                 sources: list[dict] | None = None) -> str:
    """
    Build a RAG prompt that strictly grounds the LLM in retrieved context.

    The prompt uses XML-style tags to clearly delimit context and question,
    which improves LLM accuracy and reduces hallucination.

    When `sources` is provided, each chunk is labelled with its originating
    document + page so the model can cite sources inline.
    """
    sources = sources or []

    blocks = []
    for i, chunk in enumerate(context):
        meta = sources[i] if i < len(sources) else None
        label = _source_label(meta)
        blocks.append(f"[Chunk {i+1} | Source: {label}]\n{chunk}")
    context_text = "\n\n---\n\n".join(blocks)

    instruction_block = ""
    if instruction:
        instruction_block = f"\n\n<human_instruction>\n{instruction}\n</human_instruction>"

    return f"""You are answering a question for an enterprise support system.
Use ONLY the information in the context below to answer.
Do NOT use any external knowledge.
When you use information from a chunk, cite its source inline using the
format (Source: <source label>), e.g. (Source: HR-Policy.pdf, p.4).
If you cannot cite a source for a claim, do not make the claim.

<context>
{context_text}
</context>{instruction_block}

<question>
{query}
</question>

Provide a clear, structured answer. If the context contains step-by-step instructions, present them as numbered steps. Be professional and concise."""
