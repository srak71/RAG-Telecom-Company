"""
retriever.py
------------------
Builds a unified retriever over ALL ChromaDB collections created during ingestion.

Design goals
------------
* Dynamic discovery — no hardcoded collection names. Any file dropped into data/
  and ingested via main.py automatically becomes queryable with no code changes.
* Single embedding instance — HuggingFaceEmbeddings loads the model once and is
  shared across every collection, avoiding repeated disk I/O and memory allocation.
* Per-collection k — each collection contributes k_nearest results independently,
  so a richer chroma store returns proportionally more context.
* Transparent sourcing — every returned Document carries a "collection" metadata
  key so the LLM (and any downstream logging) can see which data source matched.

Public API
----------
    build_retriever(k_nearest=3, chroma_dir=CHROMA_DIR) -> RunnableLambda

The returned callable accepts a plain query string and returns List[Document].
Plugs directly into LangChain LCEL chains via the pipe operator (|).
"""
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from config import CHROMA_DIR, EMBED_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_collection_names(chroma_dir: str) -> list[str]:
    """
    Ask ChromaDB for every collection stored at chroma_dir.

    Uses chromadb.PersistentClient (the same underlying engine used by
    langchain_chroma.Chroma) rather than scanning the filesystem, so the
    result is always authoritative and in sync with what was written during
    the ingest_*.py runs.

    Args:
        chroma_dir: Path to the persistent Chroma store — the same value
                    passed as persist_directory during ingestion.

    Returns:
        List of collection name strings (e.g. ["faq", "faq2", "telecom_guide",
        "tickets"]). Returns an empty list if no collections exist yet.
    """
    client = chromadb.PersistentClient(path=chroma_dir)
    # list_collections() returns Collection objects; we only need the name strings.
    return [col.name for col in client.list_collections()]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_retriever(
    k_nearest: int = 3,
    chroma_dir: str = CHROMA_DIR,
) -> RunnableLambda:
    """
    Build and return a unified retriever over all ChromaDB collections.

    Steps
    -----
    1. Discover every collection that exists in chroma_dir via the native client.
    2. Instantiate one shared HuggingFaceEmbeddings object (model loaded once).
    3. Wrap each collection in a Chroma vector store + LangChain retriever.
    4. Return a RunnableLambda that fans the query out to every retriever,
       annotates each result with its source collection name, and returns a
       single flat List[Document].

    Args:
        k_nearest:  Number of results fetched from each individual collection.
                    Total results ≈ k_nearest × number_of_collections.
                    Defaults to 3 (matches the original project's behaviour).
        chroma_dir: Path to the persistent Chroma store. Defaults to the
                    value in config.py so callers rarely need to override it.

    Returns:
        A RunnableLambda(str -> List[Document]) that can be used standalone
        or composed into an LCEL chain:

            retriever = build_retriever()
            docs = retriever.invoke("Why is my internet slow?")

            # Or in a chain:
            chain = retriever | format_docs | prompt | llm | StrOutputParser()

    Raises:
        RuntimeError: If no collections are found — ingestion hasn't been run yet.
    """

    # --- Step 1: Discover all collections ------------------------------------
    # We do this eagerly (at build time, not query time) so misconfiguration
    # fails loudly here rather than silently returning zero results later.
    collection_names = _discover_collection_names(chroma_dir)

    if not collection_names:
        raise RuntimeError(
            f"No ChromaDB collections found in '{chroma_dir}'.\n"
            "Run  python main.py  to ingest your data before building the retriever."
        )

    print(f"[retriever] Discovered {len(collection_names)} collection(s): {collection_names}")

    # --- Step 2: Load the embedding model ONCE --------------------------------
    # Sentence-transformers downloads and caches the model weights on first use.
    # Sharing a single HuggingFaceEmbeddings instance across all Chroma objects
    # means the model is loaded into memory only once regardless of how many
    # collections exist — important when running on CPU or a memory-limited host.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # --- Step 3: Build one LangChain retriever per collection -----------------
    # We store (name, retriever) pairs so the collection name is available
    # inside the _retrieve closure for metadata annotation.
    retrievers: list[tuple[str, object]] = []

    for name in collection_names:
        # Chroma() here connects to an EXISTING persisted collection —
        # it does NOT re-embed anything.  The embedding_function is only
        # used to embed the incoming *query* at retrieval time.
        vector_store = Chroma(
            persist_directory=chroma_dir,
            collection_name=name,
            embedding_function=embeddings,
        )

        # as_retriever() wraps the vector store in a standard LangChain
        # BaseRetriever interface. search_type="similarity" (default) ranks
        # chunks by cosine distance to the embedded query; k sets the cutoff.
        retriever = vector_store.as_retriever(search_kwargs={"k": k_nearest})
        retrievers.append((name, retriever))
        print(f"[retriever]   Attached collection '{name}' (k={k_nearest})")

    # --- Step 4: Build the merged retriever callable --------------------------
    def _retrieve(query: str) -> list[Document]:
        """
        Fan the query out to every collection and return a single flat list.

        For each collection:
          - invoke() embeds the query and fetches the top-k nearest chunks
          - we stamp "collection" onto each Document's metadata so downstream
            components (the prompt builder, logging, the LLM itself) can
            distinguish FAQ answers from resolved-ticket history from PDF
            technical reference content when formulating a response.

        Args:
            query: The raw user question string.

        Returns:
            Flat List[Document], length ≤ k_nearest × len(collection_names).
            (May be shorter if some collections have fewer than k documents.)
        """
        all_docs: list[Document] = []

        for collection_name, retriever in retrievers:
            docs = retriever.invoke(query)

            # Annotate source so the LLM and logging can attribute each chunk.
            for doc in docs:
                doc.metadata["collection"] = collection_name

            all_docs.extend(docs)

        return all_docs

    # Wrap in RunnableLambda so the retriever is a first-class LCEL component.
    # This lets it participate in chains:  retriever | next_step | ...
    return RunnableLambda(_retrieve)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How do I reset my password?"

    print(f"\n[retriever] Smoke-test query: '{query}'\n")

    retriever = build_retriever(k_nearest=3)
    results = retriever.invoke(query)

    print(f"\n[retriever] {len(results)} total result(s) returned:\n")
    for i, doc in enumerate(results, 1):
        src = doc.metadata.get("collection", "unknown")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        print(f"  [{i}] collection={src!r}")
        print(f"       {snippet!r}")
        print()
