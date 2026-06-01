"""
main.py
------------------
Entry point for the RAG Telecom ingestion pipeline.

Runs all three ingest modules in sequence. Each module scans the data/
directory for its file type and stores results in ChromaDB under a collection
named after the source filename (e.g., faq.csv → collection "faq").

Usage:
    python main.py
"""
from ingest_csv import ingest_csv
from ingest_pdf import ingest_pdf
from ingest_db import ingest_db


def main():
    print("=" * 50)
    print("RAG Telecom — Ingestion Pipeline")
    print("=" * 50 + "\n")

    ingest_csv()
    ingest_pdf()
    ingest_db()

    print("=" * 50)
    print("All ingestion complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
