"""
query.py
────────
Command-line interface for querying your RAG system.

Usage:
    python query.py                        # interactive chat loop
    python query.py -q "What is X?"       # single question
"""

import argparse

from rag_engine import build_chain, load_retriever, check_ready, _format_docs

SEPARATOR = "─" * 60


def answer_question(question: str, chain, retriever) -> None:
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    print(f"\n🤖  Answer:\n{answer}")
    print(f"\n{SEPARATOR}")
    print("📎  Sources used:")
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = src + (f" — page {int(page)+1}" if page != "" else "")
        if label not in seen:
            print(f"   • {label}")
            seen.add(label)
    print(SEPARATOR)


def interactive_loop(chain, retriever) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║      RAG — Interactive Query CLI     ║")
    print("║  Type 'exit' or 'quit' to stop       ║")
    print("╚══════════════════════════════════════╝\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("👋  Goodbye!")
            break

        answer_question(question, chain, retriever)
        print()


def main():
    parser = argparse.ArgumentParser(description="Query your local RAG system")
    parser.add_argument("-q", "--question", type=str, default=None,
                        help="Single question (skips interactive mode)")
    args = parser.parse_args()

    # Pre-flight checks
    ready, message = check_ready()
    if not ready:
        print(f"\n❌  {message}\n")
        return

    print("\n⏳  Loading retriever and chain …")
    retriever = load_retriever()
    chain = build_chain(retriever)
    print("✅  Ready!\n")

    if args.question:
        answer_question(args.question, chain, retriever)
    else:
        interactive_loop(chain, retriever)


if __name__ == "__main__":
    main()
