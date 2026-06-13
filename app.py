"""
app.py  —  Streamlit chat interface
------------------
Orchestrates the three RAG stages and renders the chat UI.

RAG pipeline (explicit three-stage flow)
-----------------------------------------
    retriever.py   RETRIEVE  → build_retriever()       query → List[Document]
    augment.py     AUGMENT   → build_prompt_inputs()   docs + question → dict
    generate.py    GENERATE  → build_generator()       dict → streamed answer

Keeping retrieval explicit (not piped inside the LLM chain) lets us:
  • Show the retrieved context in the debug expander for transparency.
  • Diagnose "I don't know" answers: if the context looks right the issue
    is in the prompt; if it looks wrong or empty the issue is in ingestion.

Usage
-----
    streamlit run app.py
"""
from dotenv import load_dotenv
import streamlit as st

from retriever import build_retriever   # R — fetch relevant chunks
from augment import build_prompt_inputs # A — format docs into context dict
from generate import build_generator    # G — prompt + LLM + output parser
from config import CHROMA_DIR


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load GROQ_API_KEY (and any other keys) from a .env file.
# No-op if the key is already present in the shell environment.
load_dotenv()


# ---------------------------------------------------------------------------
# Cached initialisation
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_rag_components():
    """
    Build and cache the retriever and generator for the whole Streamlit session.

    st.cache_resource runs once per server process — the embedding model load
    and ChromaDB connections are expensive and must not repeat on every rerun.

    Returns:
        Tuple (retriever, generator):
          retriever  — RunnableLambda: str → List[Document]
          generator  — Runnable: {"context": str, "question": str} → str stream
    """
    # R — discover all ChromaDB collections and build the fan-out retriever
    retriever = build_retriever(k_nearest=3, chroma_dir=CHROMA_DIR)

    # G — build the prompt → LLM → parser chain
    generator = build_generator()

    return retriever, generator


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Telecom Customer Care Assistant",
    page_icon="📡",
    layout="centered",
)

st.title("📡 Telecom Customer Care Assistant")
st.caption("Ask me anything about your Telecom service — billing, technical issues, plans, and more.")

# Sidebar toggle: surfaces the retrieved context chunks for each query.
# Fastest way to diagnose wrong answers — you can see exactly what the
# LLM was given as context before it responded.
show_context = st.sidebar.checkbox("🔍 Show retrieved context", value=False)

# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

# --- Render existing chat history -------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Load RAG components (cached) -------------------------------------------
retriever, generator = load_rag_components()

# --- Chat input --------------------------------------------------------------
if user_input := st.chat_input("How can I help you today?"):

    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── RETRIEVE ─────────────────────────────────────────────────────────────
    # Embed the query and fetch top-k chunks from every ChromaDB collection.
    # Done here (not inside the generator chain) so we can inspect the docs.
    docs = retriever.invoke(user_input)
    st.session_state.last_docs = docs

    # ── AUGMENT ──────────────────────────────────────────────────────────────
    # Label and join the retrieved chunks into a context string, then combine
    # with the question into the dict the generator chain expects.
    prompt_inputs = build_prompt_inputs(docs, user_input)

    # ── GENERATE ─────────────────────────────────────────────────────────────
    # Stream the answer token by token from the Groq API.
    with st.chat_message("assistant"):
        response = st.write_stream(generator.stream(prompt_inputs))

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- Debug expander ----------------------------------------------------------
# Shown below the chat when the sidebar toggle is on.
# Reveals the exact passages the retriever pulled for the last query.
# If the context looks relevant but the answer is still wrong → prompt issue.
# If the context looks unrelated or empty → ingestion / embedding issue.
if show_context and st.session_state.last_docs:
    with st.expander(
        f"🔍 Retrieved context — {len(st.session_state.last_docs)} chunks",
        expanded=True,
    ):
        for i, doc in enumerate(st.session_state.last_docs, 1):
            label = (
                doc.metadata.get("collection")
                or doc.metadata.get("source")
                or "unknown"
            )
            st.markdown(f"**[{i}] `{label}`**")
            st.text(doc.page_content[:500])
            st.divider()
