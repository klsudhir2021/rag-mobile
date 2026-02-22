"""
mobile_app.py
─────────────
Mobile-first Voice RAG Assistant (PWA-ready)
  📱 Fully responsive — works on Android & iOS
  🎤 Mic input  → Groq Whisper (speech-to-text)
  🧠 RAG        → ChromaDB + LLaMA 3.3 via Groq
  🔊 TTS output → gTTS (text-to-speech)
  📲 PWA        → installable on phone home screen
"""

import os
import io
import tempfile
import base64
from pathlib import Path

import streamlit as st
from streamlit_mic_recorder import mic_recorder   # replaces streamlit-audiorecorder
from groq import Groq
from gtts import gTTS

import config
from ingest import ingest, get_embeddings
from rag_engine import build_chain, load_retriever, check_ready

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── PWA + Mobile CSS ─────────────────────────────────────────────────────────

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="RAG Assistant">
<meta name="theme-color" content="#0F0F1A">

<style>
* { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0.5rem 1rem 5rem 1rem !important;
    max-width: 100% !important;
}
.user-bubble {
    background: linear-gradient(135deg, #6C63FF, #4ECDC4);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(108,99,255,0.3);
    word-wrap: break-word;
}
.bot-bubble {
    background: #1A1A2E;
    color: #EAEAEA;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 20% 8px 0;
    font-size: 15px;
    line-height: 1.5;
    border: 1px solid #2A2A4A;
    word-wrap: break-word;
}
.source-tag {
    font-size: 11px;
    color: #888;
    margin-top: 6px;
    padding: 3px 8px;
    background: rgba(108,99,255,0.1);
    border-radius: 10px;
    display: inline-block;
}
.status-bar {
    background: linear-gradient(135deg, #6C63FF22, #4ECDC422);
    border: 1px solid #6C63FF44;
    border-radius: 12px;
    padding: 10px 16px;
    margin: 10px 0;
    font-size: 13px;
    color: #aaa;
    text-align: center;
}
.stButton > button {
    border-radius: 50px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    width: 100% !important;
    font-size: 15px !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 14px !important;
    padding: 8px 16px !important;
}
[data-testid="stSidebar"] {
    width: 85vw !important;
    max-width: 320px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load secrets ─────────────────────────────────────────────────────────────

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
    "tts_lang":     "en",
    "auto_speak":   True,
    "last_audio_id": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_rag_system():
    with st.spinner("Loading …"):
        st.session_state.retriever  = load_retriever()
        st.session_state.chain      = build_chain(st.session_state.retriever)
        st.session_state.store_loaded = True


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Groq Whisper."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text",
            )
        return result.strip()
    finally:
        os.unlink(tmp_path)


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """Convert text to MP3 audio bytes."""
    clean = (text
        .replace("**", "").replace("*", "")
        .replace("#", "").replace("`", "")
        .replace("═", "").replace("•", "")
        .replace("[", "").replace("]", "")
    )
    tts = gTTS(text=clean[:3000], lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def autoplay_audio(audio_bytes: bytes):
    """Auto-play audio in browser."""
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay style="display:none">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )


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
        vs = Chroma(persist_directory=config.CHROMA_DIR, embedding_function=get_embeddings())
        return vs._collection.count()
    except Exception:
        return 0


def is_cloud():
    return hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets


def render_chat_bubble(role: str, content: str, sources: list = None):
    if role == "user":
        st.markdown(f'<div class="user-bubble">🧑 {content}</div>', unsafe_allow_html=True)
    else:
        src_html = ""
        if sources:
            tags = " ".join([f'<span class="source-tag">📄 {s}</span>' for s in sources[:3]])
            src_html = f'<div style="margin-top:8px">{tags}</div>'
        st.markdown(
            f'<div class="bot-bubble">🤖 {content}{src_html}</div>',
            unsafe_allow_html=True,
        )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    if not is_cloud():
        st.markdown("#### 🔑 Groq API Key")
        st.caption("Free at [console.groq.com](https://console.groq.com)")
        groq_key = st.text_input(
            "key", value=os.getenv("GROQ_API_KEY", ""),
            type="password", placeholder="gsk_...",
            label_visibility="collapsed",
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            import rag_engine
            rag_engine.GROQ_API_KEY = groq_key
    else:
        st.success("🔑 Key loaded")

    st.divider()
    st.markdown("#### 🔊 Voice")
    st.session_state.tts_lang = st.selectbox(
        "Language",
        options=["en", "hi", "es", "fr", "de", "zh", "ar", "ja"],
        format_func=lambda x: {
            "en": "🇺🇸 English", "hi": "🇮🇳 Hindi",
            "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
            "de": "🇩🇪 German",  "zh": "🇨🇳 Chinese",
            "ar": "🇸🇦 Arabic",  "ja": "🇯🇵 Japanese",
        }[x],
        label_visibility="collapsed",
    )
    st.session_state.auto_speak = st.toggle("Auto-play answers 🔊", value=True)

    st.divider()
    st.markdown("#### 📂 Documents")
    uploaded = st.file_uploader(
        "Upload", type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploaded:
        os.makedirs(config.DOCUMENTS_DIR, exist_ok=True)
        for uf in uploaded:
            with open(os.path.join(config.DOCUMENTS_DIR, uf.name), "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"✅ {len(uploaded)} file(s) saved")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Ingest", use_container_width=True, type="primary"):
            n = ingest(reset=False)
            if n:
                st.success(f"{n} chunks")
                st.session_state.store_loaded = False
                st.rerun()
    with c2:
        if st.button("🔁 Reset", use_container_width=True):
            n = ingest(reset=True)
            if n:
                st.success(f"{n} chunks")
                st.session_state.store_loaded = False
                st.rerun()

    docs = list_documents()
    if docs:
        st.caption(f"📄 {len(docs)} doc(s) · {get_store_stats()} chunks")
        for d in docs:
            ca, cb = st.columns([5, 1])
            with ca:
                st.caption(f"• {os.path.basename(d)}")
            with cb:
                if st.button("🗑", key=f"d_{d}"):
                    os.remove(d)
                    st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
---
📲 **Install as App:**
- **Android:** tap ⋮ → *Add to Home Screen*
- **iPhone:** tap Share → *Add to Home Screen*
""")

# ─── Main UI ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 10px 0 5px 0;">
    <h2 style="margin:0; font-size:22px;">🎤 RAG Assistant</h2>
    <p style="color:#888; font-size:13px; margin:4px 0 0 0;">
        Ask questions from your documents by voice or text
    </p>
</div>
""", unsafe_allow_html=True)

# API key check
if not os.environ.get("GROQ_API_KEY", "").strip():
    st.warning("🔑 Open sidebar (☰) → enter your Groq API key")
    st.stop()

# Load RAG system
if not st.session_state.store_loaded:
    ready, msg = check_ready()
    if ready:
        load_rag_system()
    elif "Vector store" in msg:
        st.markdown(
            '<div class="status-bar">👈 Open sidebar → Upload documents → Click Ingest</div>',
            unsafe_allow_html=True,
        )
        st.stop()
    else:
        st.warning(msg)
        st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_voice, tab_chat = st.tabs(["🎤 Voice", "💬 Chat History"])

with tab_voice:

    st.markdown("**🎤 Tap the mic and speak your question:**")

    # ── mic_recorder replaces audiorecorder — works on Python 3.13 ────────────
    audio = mic_recorder(
        start_prompt="⏺ Tap to Record",
        stop_prompt="⏹ Tap to Stop",
        just_once=True,          # returns audio only once per recording
        use_container_width=True,
        key="mic",
    )

    st.markdown("**Or type your question:**")
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        typed = st.text_input(
            "q", placeholder="Ask something …",
            label_visibility="collapsed",
        )
    with col_btn:
        ask = st.button("➤", use_container_width=True)

    question = None

    # Process voice — mic_recorder returns a dict with 'bytes' key
    if audio and audio.get("bytes"):
        # Avoid re-processing the same recording on reruns
        audio_id = audio.get("id", 0)
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            with st.spinner("🎙️ Transcribing …"):
                try:
                    question = transcribe_audio(audio["bytes"])
                    st.markdown(
                        f'<div class="status-bar">📝 Heard: <strong>{question}</strong></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Mic error: {e}")

    # Process text
    elif ask and typed.strip():
        question = typed.strip()

    # Generate answer
    if question:
        render_chat_bubble("user", question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🧠 Thinking …"):
            source_docs = st.session_state.retriever.invoke(question)
            answer      = st.session_state.chain.invoke(question)

        sources = []
        for doc in source_docs:
            src   = doc.metadata.get("source", "unknown")
            page  = doc.metadata.get("page", "")
            label = src + (f" p{int(page)+1}" if page != "" else "")
            if label not in sources:
                sources.append(label)

        render_chat_bubble("assistant", answer, sources)
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })

        # TTS
        with st.spinner("🔊 Generating audio …"):
            try:
                audio_bytes = text_to_speech(answer, st.session_state.tts_lang)
                st.audio(audio_bytes, format="audio/mp3")
                if st.session_state.auto_speak:
                    autoplay_audio(audio_bytes)
            except Exception as e:
                st.caption(f"TTS unavailable: {e}")

with tab_chat:
    if not st.session_state.messages:
        st.markdown(
            '<div class="status-bar">No conversation yet. Go to 🎤 Voice tab to start!</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            render_chat_bubble(msg["role"], msg["content"], msg.get("sources"))
