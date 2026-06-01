"""
ingest_csv.py
------------------
Ingests CSV files into ChromaDB for RAG retrieval.

Each CSV file gets its own Chroma collection named after its filename stem
(e.g., faq.csv → collection "faq", faq2.csv → collection "faq2").
One row = one document — no additional chunking needed.

Public API:
    ingest_csv(data_dir, chroma_dir) — call this from main.py
"""
import os
import re
import glob
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, DATA_DIR, EMBED_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_collection_name(name: str) -> str:
    """
    Convert a filename stem into a valid ChromaDB collection name.
    Rules: 3-63 chars, alphanumeric + underscores + hyphens + dots,
    must start and end with an alphanumeric character.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9._-]", "_", name)   # replace invalid chars
    name = re.sub(r"\.{2,}", ".", name)            # collapse consecutive dots
    name = name.strip("_.-")                       # strip invalid edge chars
    name = name[:63]                               # enforce max length
    if len(name) < 3:
        name = name.ljust(3, "0")                  # pad to minimum length
    return name


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def load_csv_documents(csv_path: str) -> list[Document]:
    """
    Read a CSV file and convert every row into a LangChain Document.
    Metadata includes the source filename and all column values.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        List of Document objects, one per row.
    """
    df = pd.read_csv(csv_path)
    source_name = os.path.basename(csv_path)
    docs = []

    for _, row in df.iterrows():
        content = row.to_dict()
        doc = Document(
            page_content=str(content),
            metadata={
                **{k: str(v) for k, v in content.items()},
                "source": source_name,
            },
        )
        docs.append(doc)

    return docs


def embed_and_store_documents(
    docs: list[Document],
    collection_name: str,
    chroma_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Embed documents and upsert them into a named Chroma collection.

    Args:
        docs:            Documents to embed and store.
        collection_name: Target Chroma collection (created if absent).
        chroma_dir:      Path to the persistent Chroma store.

    Returns:
        The Chroma vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = Chroma(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    vector_store.add_documents(docs)
    return vector_store


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_csv(data_dir: str = DATA_DIR, chroma_dir: str = CHROMA_DIR) -> None:
    """
    Ingest every CSV file in data_dir into its own Chroma collection.

    Collection name = sanitized filename stem (e.g., faq.csv → "faq").

    Args:
        data_dir:   Directory to scan for *.csv files.
        chroma_dir: Path to the persistent Chroma store.
    """
    print("[ingest_csv] Starting CSV ingestion...")

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"[ingest_csv] No CSV files found in '{data_dir}'.")
        return

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        collection_name = _sanitize_collection_name(Path(csv_path).stem)

        print(f"[ingest_csv]   Loading '{filename}' → collection '{collection_name}'")
        docs = load_csv_documents(csv_path)
        embed_and_store_documents(docs, collection_name, chroma_dir)
        print(f"[ingest_csv]   Stored {len(docs)} documents.")

    print("[ingest_csv] CSV ingestion complete.\n")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest_csv()
