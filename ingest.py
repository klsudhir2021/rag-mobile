"""
ingest.py
─────────
Loads every document from the /documents folder, splits them into chunks,
creates embeddings and stores everything in a local ChromaDB vector store.

Run this script once initially, and re-run it whenever you add / change files.
"""

import os
import shutil
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document as _Doc  # noqa: F401
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


# ─── Helpers ──────────────────────────────────────────────────────────────────

LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
    ".md":   TextLoader,       # Use TextLoader for markdown — no extra deps needed
}


def load_single_document(file_path: str) -> List[Document]:
    """Load a single document using the appropriate loader."""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        print(f"  ⚠️  Unsupported file type skipped: {file_path}")
        return []
    try:
        loader = loader_cls(file_path)
        docs = loader.load()
        # Tag every chunk with the source filename for easy citation
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file_path)
        return docs
    except Exception as e:
        print(f"  ❌  Error loading {file_path}: {e}")
        return []


def load_all_documents(directory: str) -> List[Document]:
    """Recursively load all supported documents from a directory."""
    all_docs: List[Document] = []
    found = False
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in config.SUPPORTED_EXTENSIONS:
                found = True
                full_path = os.path.join(root, filename)
                print(f"  📄  Loading: {filename}")
                docs = load_single_document(full_path)
                all_docs.extend(docs)
    if not found:
        print("  ⚠️  No supported documents found in the documents/ folder.")
    return all_docs


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the embedding model (downloaded on first run)."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


# ─── Main ingest pipeline ─────────────────────────────────────────────────────

def ingest(reset: bool = False) -> int:
    """
    Load documents → split → embed → store in ChromaDB.

    Parameters
    ----------
    reset : bool
        If True, wipe the existing vector store before re-indexing.

    Returns
    -------
    int
        Number of chunks indexed.
    """
    print("\n╔══════════════════════════════════════╗")
    print("║        RAG — Document Ingestion      ║")
    print("╚══════════════════════════════════════╝\n")

    # 1. Optionally wipe existing store
    if reset and os.path.exists(config.CHROMA_DIR):
        print("🗑️  Clearing existing vector store …\n")
        shutil.rmtree(config.CHROMA_DIR)

    # 2. Load documents
    print(f"📂  Scanning: {config.DOCUMENTS_DIR}\n")
    docs = load_all_documents(config.DOCUMENTS_DIR)
    if not docs:
        print("\n⚠️  No documents loaded. Add files to the documents/ folder and retry.")
        return 0
    print(f"\n✅  Loaded {len(docs)} page(s) / document(s)")

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"🔪  Split into {len(chunks)} chunk(s)  "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    # 4. Embed & store
    print(f"\n🔢  Building embeddings with '{config.EMBEDDING_MODEL}' …")
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"💾  Saved {len(chunks)} chunk(s) → {config.CHROMA_DIR}\n")
    print("✅  Ingestion complete!\n")
    return len(chunks)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the vector store before re-indexing",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
