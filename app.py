"""
app.py
──────
Streamlit web UI for the RAG system using Groq (free tier).

Run with:
    streamlit run app.py
"""

import os
import shutil
from pathlib import Path

import streamlit as st

import config
from ingest import ingest, get_embeddings
from rag_engine import build_chain, load_retriever, check_ready, GROQ_API_KEY

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📚 Local RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ─── Session state defaults ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "store_loaded" not in st.session_state:
    st.session_state.store_loaded = False


# ─── Helper functions ─────────────────────────────────────────────────────────

def load_rag_system():
    """Load retriever + chain into session state."""
    with st.spinner("⏳ Loading RAG system …"):
        st.session_state.retriever = load_retriever()
        st.session_state.chain     = build_chain(st.session_state.retriever)
        st.session_state.store_loaded = True
    st.success("✅ RAG system ready!")


def list_documents():
    """Return list of document file paths in the documents folder."""
    files = []
    for root, _, filenames in os.walk(config.DOCUMENTS_DIR):
        for f in filenames:
            if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, f))
    return files


def get_store_stats():
    """Return count of indexed chunks from ChromaDB."""
    try:
        from langchain_chroma import Chroma
        emb = get_embeddings()
        vs  = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=emb)
        return vs._collection.count()
    except Exception:
        return 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    # ── Groq API Key input ────────────────────────────────────────────────────
    st.subheader("🔑 Groq API Key")
    st.caption("Get your free key at [console.groq.com](https://console.groq.com)")

    groq_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
        label_visibility="collapsed",
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        # Patch rag_engine module so build_chain picks it up
        import rag_engine
        rag_engine.GROQ_API_KEY = groq_key_input

    st.divider()

    # ── Document uploader ──────────────────────────────────────────────────────
    st.subheader("📂 Upload Documents")
    st.caption(f"Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}")

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        saved = []
        for uf in uploaded_files:
            dest = os.path.join(config.DOCUMENTS_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        st.success(f"✅ Saved {len(saved)} file(s) to documents/")

    st.divider()

    # ── Ingestion controls ────────────────────────────────────────────────────
    st.subheader("🔄 Index Documents")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Ingest", use_container_width=True, type="primary"):
            with st.spinner("Indexing …"):
                n = ingest(reset=False)
            if n:
                st.success(f"Indexed {n} chunks")
                st.session_state.store_loaded = False
                st.rerun()
            else:
                st.warning("No documents found.")

    with col2:
        if st.button("🔁 Re-index", use_container_width=True):
            with st.spinner("Re-indexing …"):
                n = ingest(reset=True)
            if n:
                st.success(f"Re-indexed {n} chunks")
                st.session_state.store_loaded = False
                st.rerun()

    st.divider()

    # ── Document list ─────────────────────────────────────────────────────────
    st.subheader("📋 Indexed Documents")
    docs = list_documents()
    if docs:
        for d in docs:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.caption(f"📄 {os.path.basename(d)}")
            with col_b:
                if st.button("🗑️", key=f"del_{d}", help="Delete this file"):
                    os.remove(d)
                    st.rerun()
        chunks_count = get_store_stats()
        st.caption(f"Total chunks in store: **{chunks_count}**")
    else:
        st.caption("No documents yet. Upload some files above!")

    st.divider()

    # ── Model info ────────────────────────────────────────────────────────────
    st.subheader("🤖 Model Info")
    import rag_engine
    st.caption(f"LLM: `{rag_engine.GROQ_MODEL}`")
    st.caption(f"Embeddings: `all-MiniLM-L6-v2`")
    st.caption(f"Vector Store: `ChromaDB (local)`")

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────

st.title("📚 Local RAG Assistant")
st.caption("Ask questions about your uploaded documents — powered by Groq (free).")

# Status checks
import rag_engine as _re
groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())

if not groq_key_set:
    st.warning("🔑 Enter your **Groq API key** in the sidebar to get started. "
               "Get one free at https://console.groq.com")
    st.stop()

# Auto-load chain if store is available and not yet loaded
if not st.session_state.store_loaded:
    ready, msg = check_ready()
    if ready:
        load_rag_system()
    elif "Vector store" in msg:
        st.info("👈  Upload documents and click **▶ Ingest** to get started.")
    else:
        st.warning(msg)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for s in message["sources"]:
                    st.caption(f"• {s}")

# Chat input
if question := st.chat_input("Ask a question about your documents …"):

    if not st.session_state.store_loaded:
        st.error("⚠️ Please ingest documents first using the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            source_docs = st.session_state.retriever.invoke(question)
            answer      = st.session_state.chain.invoke(question)

        st.markdown(answer)

        # Sources
        seen_sources = []
        for doc in source_docs:
            src   = doc.metadata.get("source", "unknown")
            page  = doc.metadata.get("page", "")
            label = src + (f" — page {int(page)+1}" if page != "" else "")
            if label not in seen_sources:
                seen_sources.append(label)

        if seen_sources:
            with st.expander("📎 Sources", expanded=False):
                for s in seen_sources:
                    st.caption(f"• {s}")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": seen_sources,
    })