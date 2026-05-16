"""
app.py — Command-line interface for the Legal Document RAG Assistant
Usage:
    python app.py --ingest          # First time: load docs & build index
    python app.py                   # Chat mode (index already built)
"""

import os
import sys
import argparse
from pathlib import Path

from rag_engine import load_documents, build_chunks, build_index, load_index, ask

DOCS_FOLDER = "documents"
INDEX_PATH  = "data/index.faiss"
BANNER = """
╔══════════════════════════════════════════════════════╗
║       Legal Document RAG Assistant  ⚖️               ║
║       Powered by Groq (LLaMA 3) + FAISS              ║
╚══════════════════════════════════════════════════════╝
Type your legal question and press Enter.
Commands: 'sources' = show last sources | 'quit' = exit
"""


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        key = input("Enter your Groq API key: ").strip()
        if not key:
            print("[✗] No API key provided. Get one free at https://console.groq.com")
            sys.exit(1)
    return key


def ingest_documents():
    """Load documents, chunk, embed, and save index."""
    docs_path = Path(DOCS_FOLDER)
    if not docs_path.exists():
        docs_path.mkdir()
        print(f"[!] Created '{DOCS_FOLDER}/' folder.")
        print(f"[!] Add your legal PDF or TXT files to '{DOCS_FOLDER}/' then run again with --ingest")
        sys.exit(0)

    documents = load_documents(DOCS_FOLDER)
    chunks = build_chunks(documents)
    build_index(chunks, save=True)
    print("\n[✓] Ingestion complete! Run 'python app.py' to start chatting.\n")


def chat_mode(api_key: str):
    """Interactive chat loop."""
    if not Path(INDEX_PATH).exists():
        print("[!] No index found. Run first with: python app.py --ingest")
        sys.exit(1)

    index, embedder, chunks = load_index()
    print(BANNER)

    last_result = None
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Bye!]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("[Bye!]")
            break
        if query.lower() == "sources" and last_result:
            print("\n── Sources used in last answer ──")
            for i, s in enumerate(last_result["sources"], 1):
                print(f"  [{i}] {s['source']} (score: {s['score']})")
                print(f"      …{s['excerpt']}")
            continue

        print("\n[Retrieving & generating…]")
        result = ask(query, index, embedder, chunks, api_key)
        last_result = result

        print(f"\nAssistant:\n{result['answer']}")
        print(f"\n── {len(result['sources'])} source(s) used — type 'sources' to see them ──")


def main():
    parser = argparse.ArgumentParser(description="Legal Document RAG Assistant")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents and build vector index")
    args = parser.parse_args()

    if args.ingest:
        ingest_documents()
    else:
        api_key = get_api_key()
        chat_mode(api_key)


if __name__ == "__main__":
    main()
