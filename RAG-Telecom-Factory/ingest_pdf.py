"""
ingest_pdf.py
------------------
Ingests PDF files into ChromaDB for RAG retrieval.

Each PDF file gets its own Chroma collection named after its filename stem
(e.g., telecom_guide.pdf → collection "telecom_guide").
PDFs are chunked using RecursiveCharacterTextSplitter with:
    chunk_size    = 500  characters
    chunk_overlap = 125  characters (25% overlap for context continuity)

Public API:
    ingest_pdf(data_dir, chroma_dir) — call this from main.py
"""
import os
import re
import glob
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, DATA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


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

def load_and_chunk_pdf(pdf_path: str) -> list:
    """
    Load a PDF and split it into overlapping text chunks.

    Uses PyPDFLoader to extract text page-by-page, then
    RecursiveCharacterTextSplitter to create fixed-size chunks with overlap.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document chunk objects.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def embed_and_store_documents(
    docs: list,
    collection_name: str,
    chroma_dir: str = CHROMA_DIR,
) -> Chroma:
    """
    Embed document chunks and upsert them into a named Chroma collection.

    Args:
        docs:            Document chunks to embed and store.
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

def ingest_pdf(data_dir: str = DATA_DIR, chroma_dir: str = CHROMA_DIR) -> None:
    """
    Ingest every PDF file in data_dir into its own Chroma collection.

    Collection name = sanitized filename stem (e.g., telecom_guide.pdf → "telecom_guide").

    Args:
        data_dir:   Directory to scan for *.pdf files.
        chroma_dir: Path to the persistent Chroma store.
    """
    print("[ingest_pdf] Starting PDF ingestion...")

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"[ingest_pdf] No PDF files found in '{data_dir}'.")
        return

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        collection_name = _sanitize_collection_name(Path(pdf_path).stem)

        print(f"[ingest_pdf]   Loading '{filename}' → collection '{collection_name}'")
        chunks = load_and_chunk_pdf(pdf_path)
        embed_and_store_documents(chunks, collection_name, chroma_dir)
        print(f"[ingest_pdf]   Stored {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("[ingest_pdf] PDF ingestion complete.\n")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest_pdf()
