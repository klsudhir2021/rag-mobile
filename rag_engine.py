"""
rag_engine.py
─────────────
Improved RAG engine with better accuracy:
  ✅ Stricter prompt — model cannot use outside knowledge
  ✅ MMR retrieval — reduces duplicate chunks, improves diversity
  ✅ Larger context window passed to LLM
  ✅ Confidence check — model must say when it doesn't know
  ✅ Better embedding model (all-mpnet-base-v2)
"""

import os
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import config

# ─── Groq settings ────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Available free Groq models:
#   "llama-3.3-70b-versatile"   → best quality ✅ recommended
#   "llama-3.1-8b-instant"      → fastest, use if speed is priority
#   "mixtral-8x7b-32768"        → large context window (32k tokens)
#   "gemma2-9b-it"              → Google Gemma 2
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Improved Prompt ──────────────────────────────────────────────────────────
# Key improvements:
#  1. Explicitly forbids using outside/pretrained knowledge
#  2. Requires verbatim grounding ("based on the text above")
#  3. Forces model to admit uncertainty rather than hallucinate
#  4. Asks for direct quotes when possible

SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to \
answer questions using the document excerpts provided below in the [CONTEXT] section.

STRICT RULES — follow these exactly:
1. ONLY use information explicitly stated in the [CONTEXT]. Do NOT use any outside \
knowledge, assumptions, or information from your training data.
2. If the [CONTEXT] does not contain enough information to answer the question, \
respond EXACTLY with:
   "The provided documents do not contain enough information to answer this question."
   Do NOT guess, infer, or fill gaps with general knowledge.
3. When answering, reference the source document name (shown in each [Source] tag).
4. If you quote text directly from the context, use quotation marks.
5. Keep your answer factual, concise, and strictly grounded in the context.
6. If multiple sources say different things, mention both and note the discrepancy.

[CONTEXT]
{context}
"""

HUMAN_PROMPT = "Question: {question}"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Format retrieved chunks with clear source labels."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        label  = source + (f", page {int(page) + 1}" if page != "" else "")
        parts.append(
            f"[Source {i} — {label}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n═══════════════════════\n\n".join(parts)


def _is_store_ready() -> bool:
    return os.path.exists(config.CHROMA_DIR) and bool(os.listdir(config.CHROMA_DIR))


# ─── Public API ───────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return embedding model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def load_retriever():
    """
    Load ChromaDB and return an MMR retriever.

    MMR (Maximal Marginal Relevance) fetches a larger candidate pool
    then re-ranks to balance relevance AND diversity, so the LLM gets
    a wider spread of information rather than near-duplicate chunks.
    """
    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(
        search_type="mmr",                      # ← MMR instead of plain similarity
        search_kwargs={
            "k":           config.TOP_K,        # chunks sent to LLM
            "fetch_k":     config.FETCH_K,      # candidates before re-ranking
            "lambda_mult": 0.7,                 # 1.0 = pure relevance, 0.0 = pure diversity
        },
    )


def build_chain(retriever):
    """Build and return the improved RAG chain."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,        # 0 = deterministic, no creative hallucination
        max_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    chain = (
        {
            "context":  retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_answer(question: str, retriever, chain) -> Tuple[str, List[Document]]:
    """Run the RAG chain and return (answer, source_documents)."""
    source_docs = retriever.invoke(question)
    answer      = chain.invoke(question)
    return answer, source_docs


def check_ready() -> Tuple[bool, str]:
    """Check whether the system is ready to answer questions."""
    key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not key:
        return False, (
            "GROQ_API_KEY is not set.\n"
            "Get your free key at https://console.groq.com and add it to your .env file."
        )
    if not _is_store_ready():
        return False, "Vector store not found. Please run `python ingest.py` first."
    return True, "Ready"