"""
config.py
------------------
Global configuration constants shared across all ingest and RAG pipeline files.

Collection names are NOT defined here — they are derived dynamically from the
source filename stem (e.g., faq.csv → collection "faq") inside each ingest module.
"""

CHROMA_DIR = "chroma_store"
DATA_DIR = "data"

# Embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings (used by ingest_pdf.py)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 125
