"""
ingest_db.py
------------------
Ingests SQLite ticket databases into ChromaDB for RAG retrieval.

Each .db file gets its own Chroma collection named after its filename stem
(e.g., tickets.db → collection "tickets").
Only resolved tickets are indexed; issue description + resolution are combined
into a single searchable text block per row — no additional chunking needed.

Public API:
    ingest_db(data_dir, chroma_dir) — call this from main.py
"""
import os
import re
import glob
import sqlite3
from pathlib import Path

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
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    name = name.strip("_.-")
    name = name[:63]
    if len(name) < 3:
        name = name.ljust(3, "0")
    return name


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def load_db_documents(db_path: str) -> list[Document]:
    """
    Load all resolved tickets from a SQLite database and convert them into
    LangChain Documents.  Issue description and resolution are merged into a
    single searchable text block.

    Args:
        db_path: Absolute or relative path to the SQLite file.

    Returns:
        List of Document objects, one per resolved ticket.
    """
    docs = []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()
    conn.close()

    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}\n"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": os.path.basename(db_path),
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "status": row["status"],
                },
            )
        )

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

def ingest_db(data_dir: str = DATA_DIR, chroma_dir: str = CHROMA_DIR) -> None:
    """
    Ingest every SQLite (.db) file in data_dir into its own Chroma collection.

    Collection name = sanitized filename stem (e.g., tickets.db → "tickets").

    Args:
        data_dir:   Directory to scan for *.db files.
        chroma_dir: Path to the persistent Chroma store.
    """
    print("[ingest_db] Starting database ingestion...")

    db_files = glob.glob(os.path.join(data_dir, "*.db"))
    if not db_files:
        print(f"[ingest_db] No .db files found in '{data_dir}'.")
        return

    for db_path in db_files:
        filename = os.path.basename(db_path)
        collection_name = _sanitize_collection_name(Path(db_path).stem)

        print(f"[ingest_db]   Loading '{filename}' → collection '{collection_name}'")
        docs = load_db_documents(db_path)
        embed_and_store_documents(docs, collection_name, chroma_dir)
        print(f"[ingest_db]   Stored {len(docs)} resolved ticket documents.")

    print("[ingest_db] Database ingestion complete.\n")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest_db()
