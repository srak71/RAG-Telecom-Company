"""
generate.py  —  Generation step (the "G" in RAG)
------------------
Builds the LLM chain that takes an augmented prompt input and generates
a streamed answer using the Groq API.

Receives the output of augment.py's build_prompt_inputs() — a dict with
"context" and "question" keys — and streams the LLM's response.

Chain
-----
    {"context": str, "question": str}
        │
        ▼
    ChatPromptTemplate      ← injects context + question into system/human messages
        │
        ▼
    ChatGroq                ← calls Groq API, streams tokens
        │
        ▼
    StrOutputParser         ← yields plain text chunks

Public API
----------
    build_generator(model_name, temperature) → Runnable
        The returned Runnable accepts {"context": str, "question": str}
        and can be called with .invoke() or .stream().
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Default model — fast and capable on Groq's free tier.
# Override by passing model_name to build_generator() if needed.
DEFAULT_MODEL = "qwen/qwen3-32b"

# System prompt that defines the assistant's role and grounding behaviour.
# Kept as a module-level constant so it can be read/adjusted without
# touching the chain-building logic.
SYSTEM_PROMPT = (
    "You are a helpful Telecom customer care assistant. "
    "Use the context passages below to answer the customer's question. "
    "The context comes from three sources: an FAQ database, resolved support "
    "tickets, and a technical reference guide. "
    "Base your answer on the context provided. "
    "If the context does not contain enough information to answer fully, "
    "say so politely and suggest the customer contact support for further help. "
    "Be concise, friendly, and professional."
)


def build_generator(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
):
    """
    Build and return the generation chain: prompt → LLM → output parser.

    This chain is the "G" in RAG.  It takes the structured dict produced by
    augment.build_prompt_inputs() and streams a plain-text answer.

    The chain is intentionally kept separate from retrieval and augmentation
    so each stage can be swapped, tested, or logged independently.

    Args:
        model_name:  Groq model identifier.  Defaults to llama-3.3-70b-versatile.
        temperature: Sampling temperature.  0.0 = deterministic / factual.

    Returns:
        A LangChain Runnable that accepts {"context": str, "question": str}
        and returns / streams a str answer.

    Note:
        GROQ_API_KEY must be set in the environment (or in a .env file loaded
        via python-dotenv) before calling this function.
    """

    # --- LLM -----------------------------------------------------------------
    # ChatGroq reads GROQ_API_KEY automatically from os.environ.
    # temperature=0 keeps answers consistent and avoids hallucination drift.
    llm = ChatGroq(model=model_name, temperature=temperature)

    # --- Prompt template -----------------------------------------------------
    # Two-message structure:
    #   system  — sets the assistant persona and grounding rule
    #   human   — carries the retrieved context block + the user's question
    #
    # {context} and {question} are filled by build_prompt_inputs() in augment.py.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\nCustomer question: {question}",
        ),
    ])

    # --- Chain ---------------------------------------------------------------
    # prompt  → formats the dict into a list of chat messages
    # llm     → calls Groq API and returns an AIMessage (streamable)
    # parser  → extracts the plain string content from the AIMessage
    chain = prompt | llm | StrOutputParser()

    return chain
