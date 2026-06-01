# RAG Telecom Customer Care Assistant

A Retrieval-Augmented Generation (RAG) chatbot for Telecom customer care. The assistant answers customer questions by retrieving relevant context from an FAQ database, resolved support tickets, and a technical reference guide — then generating a grounded response via a Groq LLM.

## Contents

| File | Purpose |
|------|---------|
| `config.py` | Shared constants (paths, embedding model, chunk settings) |
| `main.py` | Ingestion pipeline orchestrator — run this first |
| `ingest_csv.py` | Ingests FAQ CSV files into ChromaDB |
| `ingest_pdf.py` | Ingests Technical Reference Guide PDF into ChromaDB |
| `ingest_db.py` | Ingests resolved support tickets from SQLite into ChromaDB |
| `retriever.py` | **R**etrieve — discovers all collections, fans out query, returns chunks |
| `augment.py` | **A**ugment — labels and joins retrieved chunks into a context string |
| `generate.py` | **G**enerate — prompt template + Groq LLM + output parser chain |
| `app.py` | Streamlit chat interface — orchestrates R → A → G |
| `data/` | Source data files (CSV, PDF, SQLite) |
| `chroma_store/` | Persisted ChromaDB vector store (created by `main.py`) |

## Setup

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Clone the repository

```bash
git clone <repo-url>
cd RAG-Telecom-Project
```

### 2. Create and activate a virtual environment

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -e .
```

### 3. Configure API keys

Copy `.env.sample` to `.env` and fill in your API keys:

```bash
cp .env.sample .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

- **Groq API key**: Get from [Groq Console](https://console.groq.com/)
- **LangSmith API key**: Get from [LangSmith](https://smith.langchain.com/) (optional, for tracing)

### 4. Run the ingestion pipeline

Ingest all data sources into ChromaDB. This must be run before starting the app:

```bash
python main.py
```

This scans `data/` and creates one ChromaDB collection per file:

| File | Collection |
|------|-----------|
| `faq.csv` | `faq` |
| `faq2.csv` | `faq2` |
| `telecom_guide.pdf` | `telecom_guide` |
| `tickets.db` | `tickets` |

### 5. Launch the chatbot

```bash
streamlit run app.py
```

## Getting Started

1. Add your data files to the `data/` directory (CSV, PDF, or SQLite `.db`).
2. Run `python main.py` to ingest them into ChromaDB.
3. Run `streamlit run app.py` to start the chat interface.
4. Enable **🔍 Show retrieved context** in the sidebar to see exactly which passages the retriever is pulling for each query — useful for debugging answers.

## Notes

- **Adding new data**: drop any new `.csv`, `.pdf`, or `.db` file into `data/` and re-run `python main.py`. The retriever discovers collections dynamically — no code changes needed.
- **Changing the model**: edit `DEFAULT_MODEL` in `generate.py`. Any model available on [Groq](https://console.groq.com/docs/models) works.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace — used for both ingestion and retrieval.
- Each ingest module can also be run standalone: `python ingest_csv.py`, `python ingest_pdf.py`, `python ingest_db.py`.
