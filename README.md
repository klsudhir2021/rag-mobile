# 📱 Mobile RAG Assistant

A mobile-first, voice-enabled RAG app installable as a PWA on Android & iOS.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run mobile_app.py
```

## 📲 Install on Phone (PWA)

1. Deploy to Streamlit Cloud (see below)
2. Open the URL in your phone browser
3. **Android:** tap ⋮ menu → "Add to Home Screen"
4. **iPhone:** tap Share icon → "Add to Home Screen"

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Go to https://share.streamlit.io
3. Select repo → set main file to `mobile_app.py`
4. Add secret: `GROQ_API_KEY = "gsk_your_key"`
5. Deploy → share the URL!

## 📁 Required Files (copy from your RAG project)

Place these files in the same folder as `mobile_app.py`:
- `config.py`
- `ingest.py`
- `rag_engine.py`
- `documents/` folder

## 🔑 .streamlit/secrets.toml (local)

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```
