"""
augment.py  —  Augmentation step (the "A" in RAG)
------------------
Takes the raw retrieved Documents and formats them into the structured
context string that gets injected into the LLM prompt.

This is the bridge between retrieval and generation:
  List[Document]  →  {"context": str, "question": str}

The output dict is passed directly to the generate.py chain.

Public API
----------
    format_docs(docs)               → str   (context string only)
    build_prompt_inputs(docs, q)    → dict  (ready for the LLM chain)
"""
from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Join retrieved Document chunks into a single labeled context string.

    Each chunk is prefixed with its source collection name (e.g. [faq],
    [tickets], [telecom_guide]) so the LLM knows where each passage came
    from and can reference it when formulating an answer.

    A clear "(no relevant context found)" placeholder is returned when the
    retriever comes back empty — this prevents the LLM from hallucinating
    and signals the prompt to fall back to its "I don't know" instruction.

    Args:
        docs: List[Document] as returned by retriever.py's build_retriever().
              Each Document is expected to have a "collection" key in its
              metadata (stamped on by retriever.py).

    Returns:
        Multi-line string ready to be injected as {context} in the prompt.
    """
    if not docs:
        return "(no relevant context found)"

    parts = []
    for doc in docs:
        # "collection" is stamped onto metadata by retriever.py so we know
        # which data source (faq, tickets, telecom_guide, …) this chunk came from.
        # Fall back to "source" (set during ingestion) or a plain label if absent.
        label = (
            doc.metadata.get("collection")
            or doc.metadata.get("source")
            or "context"
        )
        parts.append(f"[{label}]\n{doc.page_content}")

    return "\n\n".join(parts)


def build_prompt_inputs(docs: list[Document], question: str) -> dict:
    """
    Combine retrieved documents and the user question into a single dict
    that maps directly to the {context} and {question} slots in the
    generate.py prompt template.

    This is the complete augmentation step — it takes the two inputs that
    are available at query time (what we retrieved, what the user asked)
    and produces the single structured object the LLM chain consumes.

    Args:
        docs:     Retrieved documents from build_retriever().invoke(question).
        question: The raw user question string.

    Returns:
        {"context": str, "question": str}
    """
    return {
        "context": format_docs(docs),
        "question": question,
    }
