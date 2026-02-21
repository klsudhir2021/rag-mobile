"""
voice_app.py
────────────
Voice-enabled RAG Assistant:
  🎤 Mic input  → Groq Whisper (speech-to-text, free)
  🧠 RAG        → ChromaDB + LLM (answer from documents)
  🔊 TTS output → gTTS (text-to-speech, free)

Run with:
    streamlit run voice_app.py
"""

import os
import io
import time
import tempfile
from pathlib import Path

import streamlit as st
from audiorecorder import audiorecorder           # pip install streamlit-audiorecorder
from groq import Groq                             # pip install groq
from gtts import gTTS                             # pip install gtts
import base64

import config
from ingest import ingest, get_embeddings
from rag_engine import build_chain, load_retriever, check_ready

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎤 Voice RAG Assistant",
    page_icon="🎤",
    layout="wide",
)

# ─── Load secrets (Streamlit Cloud compatible) ────────────────────────────────

if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    import rag_engine
    rag_engine.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ─── Session state ────────────────────────────────────────────────────────────

for k, v in {
    "messages":     [],
    "chain":        None,
    "retriever":    None,
    "store_loaded": False,
    "last_answer":  "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helper functions ─────────────────────────────────────────────────────────

def load_rag_system():
    with st.spinner("⏳ Loading RAG system …"):
        st.session_state.retriever  = load_retriever()
        st.session_state.chain      = build_chain(st.session_state.retriever)
        st.session_state.store_loaded = True


def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to Groq Whisper for free speech-to-text."""
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",   # fast & free on Groq
                file=audio_file,
                response_format="text",
            )
        return transcription.strip()
    finally:
        os.unlink(tmp_path)


def text_to_speech(text: str) -> bytes:
    """Convert text to speech using gTTS and return audio bytes."""
    # Clean text for TTS — remove markdown symbols
    clean = (text
        .replace("**", "").replace("*", "")
        .replace("#", "").replace("`", "")
        .replace("═", "").replace("•", "")
    )
    tts = gTTS(text=clean, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def autoplay_audio(audio_bytes: bytes):
    """Embed audio in page and autoplay it."""
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(html, unsafe_allow_html=True)


def list_documents():
    files = []
    for root, _, filenames in os.walk(config.DOCUMENTS_DIR):
        for f in filenames:
            if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, f))
    return files


def get_store_stats():
    try:
        from langchain_chroma import Chroma
        vs = Chroma(
            persist_directory=config.CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
        return vs._collection.count()
    except Exception:
        return 0


def is_cloud():
    return hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    # API Key
    if not is_cloud():
        st.subheader("🔑 Groq API Key")
        st.caption("Free at [console.groq.com](https://console.groq.com)")
        groq_key_input = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            placeholder="gsk_xxxxxxxxxxxxxxxxxxxx",
            label_visibility="collapsed",
        )
        if groq_key_input:
            os.environ["GROQ_API_KEY"] = groq_key_input
            import rag_engine
            rag_engine.GROQ_API_KEY = groq_key_input
    else:
        st.success("🔑 API key loaded from secrets")

    st.divider()

    # TTS Language
    st.subheader("🔊 Voice Settings")
    tts_lang = st.selectbox(
        "Speech Language",
        options=["en", "hi", "es", "fr", "de", "zh", "ar", "ja"],
        format_func=lambda x: {
            "en": "🇺🇸 English", "hi": "🇮🇳 Hindi",
            "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
            "de": "🇩🇪 German",  "zh": "🇨🇳 Chinese",
            "ar": "🇸🇦 Arabic",  "ja": "🇯🇵 Japanese",
        }.get(x, x),
    )
    auto_speak = st.toggle("🔊 Auto-play response", value=True)

    st.divider()

    # Document upload
    st.subheader("📂 Upload Documents")
    st.caption(f"Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
        saved = []
        for uf in uploaded_files:
            dest = os.path.join(config.DOCUMENTS_DIR, uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved.append(uf.name)
        if saved:
            st.success(f"✅ Saved {len(saved)} file(s)")

    st.divider()

    # Ingestion
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

    # Document list
    st.subheader("📋 Indexed Documents")
    docs = list_documents()
    if docs:
        for d in docs:
            ca, cb = st.columns([4, 1])
            with ca:
                st.caption(f"📄 {os.path.basename(d)}")
            with cb:
                if st.button("🗑️", key=f"del_{d}"):
                    os.remove(d)
                    st.rerun()
        st.caption(f"Total chunks: **{get_store_stats()}**")
    else:
        st.caption("No documents yet.")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Main area ────────────────────────────────────────────────────────────────

st.title("🎤 Voice RAG Assistant")
st.caption("Speak your question — get an answer from your documents, read aloud.")

# API key check
groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
if not groq_key_set:
    st.warning("🔑 Enter your **Groq API key** in the sidebar. Free at https://console.groq.com")
    st.stop()

# Auto-load RAG system
if not st.session_state.store_loaded:
    ready, msg = check_ready()
    if ready:
        load_rag_system()
    elif "Vector store" in msg:
        st.info("👈 Upload documents and click **▶ Ingest** to get started.")
        st.stop()
    else:
        st.warning(msg)
        st.stop()

# ── Voice Input Section ───────────────────────────────────────────────────────

st.markdown("### 🎤 Ask Your Question")

col_mic, col_type = st.columns([1, 1])

with col_mic:
    st.markdown("**Option A — Speak:**")
    audio = audiorecorder(
        start_prompt="⏺ Click to Record",
        stop_prompt="⏹ Click to Stop",
        pause_prompt="",
        key="audio_recorder",
    )

with col_type:
    st.markdown("**Option B — Type:**")
    typed_question = st.text_input(
        "Type your question",
        placeholder="e.g. What does the document say about …?",
        label_visibility="collapsed",
    )
    ask_btn = st.button("💬 Ask", type="primary", use_container_width=True)

st.divider()

# ── Process input ─────────────────────────────────────────────────────────────

question = None

# Voice input
if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    with st.spinner("🎙️ Transcribing your speech …"):
        try:
            question = transcribe_audio(audio.export().read())
            st.success(f"📝 You said: **{question}**")
        except Exception as e:
            st.error(f"Transcription failed: {e}")

# Text input
elif ask_btn and typed_question.strip():
    question = typed_question.strip()

# ── Generate & speak answer ───────────────────────────────────────────────────

if question:
    # Add to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("🧠 Searching documents and generating answer …"):
        source_docs = st.session_state.retriever.invoke(question)
        answer      = st.session_state.chain.invoke(question)

    st.session_state.last_answer = answer
    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": list({
            doc.metadata.get("source", "unknown") +
            (f" — page {int(doc.metadata['page'])+1}" if doc.metadata.get("page", "") != "" else "")
            for doc in source_docs
        }),
    })

    # Display answer
    st.markdown("### 🤖 Answer")
    st.markdown(answer)

    # Sources
    seen = []
    for doc in source_docs:
        src   = doc.metadata.get("source", "unknown")
        page  = doc.metadata.get("page", "")
        label = src + (f" — page {int(page)+1}" if page != "" else "")
        if label not in seen:
            seen.append(label)
    if seen:
        with st.expander("📎 Sources", expanded=False):
            for s in seen:
                st.caption(f"• {s}")

    # Text-to-speech
    st.markdown("### 🔊 Listen to Answer")
    with st.spinner("🔊 Generating speech …"):
        try:
            audio_bytes = text_to_speech(answer)
            st.audio(audio_bytes, format="audio/mp3")
            if auto_speak:
                autoplay_audio(audio_bytes)
        except Exception as e:
            st.warning(f"TTS failed: {e}")

    st.divider()

# ── Chat History ──────────────────────────────────────────────────────────────

if st.session_state.messages:
    st.markdown("### 💬 Conversation History")
    for msg in reversed(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")