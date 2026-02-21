import os
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR   = os.path.join(BASE_DIR, "documents")
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma_db")

# ─── Embedding model ──────────────────────────────────────────────────────────
# Upgraded from all-MiniLM-L6-v2 → better semantic understanding & accuracy
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ─── Chunking ─────────────────────────────────────────────────────────────────
# Larger chunks = more context per retrieved piece → fewer missing details
# More overlap = less chance of cutting a sentence/idea at a boundary
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# ─── Retrieval ────────────────────────────────────────────────────────────────
# Retrieve more candidates, then re-rank for precision
TOP_K           = 6    # chunks passed to LLM
FETCH_K         = 20   # candidates fetched before MMR re-ranking

# ─── Groq ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")

# ─── Supported file extensions ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".md"]