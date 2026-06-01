We are building a traditional Generative AI RAG (Retrieval Augmented Generation) customer care assistant for Telecom. This project will be created in two stages, ingestion and RAG.

We will be using LangChain, Python, Groq as the LLM, ChromaDB as the vector database, Huggingface sentence transformers (all-MiniLM-L6-v2) for embedding, and Streamlit for the interface.

For this project, always show all your thinking.

--------------------------------------------------------------------------------------------------------------------

File structure:

    config.py          — shared constants (paths, embedding model, chunk settings)
    ingest_csv.py      — CSV ingestion; exposes ingest_csv()
    ingest_pdf.py      — PDF ingestion; exposes ingest_pdf()
    ingest_db.py       — SQLite ingestion; exposes ingest_db()
    main.py            — orchestrator; calls all three ingest functions in sequence

Run the full ingestion pipeline with:

    python main.py

--------------------------------------------------------------------------------------------------------------------

Step 1: Indexing

All ingest modules follow the same pattern:
- Scan data/ for files of the relevant type (*.csv, *.pdf, *.db)
- For each file, derive the ChromaDB collection name from the filename stem
  (e.g., faq.csv → collection "faq", telecom_guide.pdf → collection "telecom_guide")
- Embed using sentence-transformers/all-MiniLM-L6-v2 via HuggingFaceEmbeddings
- Persist to chroma_store/

Collection names are NOT hardcoded in config.py — they are derived dynamically at
runtime so that dropping a new file into data/ automatically creates its own collection.

Each module also runs standalone:
    python ingest_csv.py
    python ingest_pdf.py
    python ingest_db.py


1. [DATABASE] ingest_db.py — Customer tickets (SQLite)

Source: data/*.db
Table: tickets, filtered to WHERE status = 'resolved'
Schema columns used: ticket_id, category, issue_type, description, resolution, status
Document format: issue_type + description + resolution merged into one text block
Chunking: none (one document per resolved ticket row)
Public function: ingest_db(data_dir, chroma_dir)


2. [CSV] ingest_csv.py — FAQs

Source: data/*.csv
Document format: each row converted to a dict string (one document per row)
Chunking: none (one document per row)
Public function: ingest_csv(data_dir, chroma_dir)


3. [PDF] ingest_pdf.py — Technical Reference Guide

Source: data/*.pdf
Loader: PyPDFLoader (langchain_community)
Chunking: RecursiveCharacterTextSplitter
    chunk_size    = 500 characters
    chunk_overlap = 125 characters (25% overlap)
Public function: ingest_pdf(data_dir, chroma_dir)

--------------------------------------------------------------------------------------------------------------------

Step 2: Retrieval Augmented Generation

Use the same embedding method (sentence-transformers/all-MiniLM-L6-v2) to embed the
user query, retrieve the most relevant chunks from ChromaDB, augment the prompt with
both the query and retrieved chunks, then call the Groq LLM API to generate the response.

--------------------------------------------------------------------------------------------------------------------
