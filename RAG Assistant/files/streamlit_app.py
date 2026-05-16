import os, tempfile
from pathlib import Path
import streamlit as st
from rag_engine import extract_text_from_pdf, extract_text_from_txt, build_chunks, build_index, ask

st.set_page_config(page_title="Legal RAG Assistant", page_icon="⚖️", layout="wide")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
    st.session_state.index = None
    st.session_state.embedder = None
    st.session_state.chunks = None
    st.session_state.history = []

with st.sidebar:
    st.title("⚖️ Legal RAG Assistant")
    st.write("Upload legal documents, then ask questions.")
    st.divider()
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"], accept_multiple_files=True)

    if st.button("Build Index", use_container_width=True, type="primary"):
        if not files:
            st.error("Upload at least one file.")
        else:
            with st.spinner("Building index..."):
                try:
                    docs = []
                    for f in files:
                        suffix = Path(f.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        text = extract_text_from_pdf(tmp_path) if suffix == ".pdf" else extract_text_from_txt(tmp_path)
                        docs.append({"text": text, "source": f.name})
                        os.unlink(tmp_path)
                    chunks = build_chunks(docs)
                    idx, emb, chunks = build_index(chunks)
                    st.session_state.index = idx
                    st.session_state.embedder = emb
                    st.session_state.chunks = chunks
                    st.session_state.indexed = True
                    st.session_state.history = []
                    st.success(f"Ready! {len(chunks)} chunks indexed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.success("Index ready!") if st.session_state.indexed else st.info("Upload docs and build index.")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

st.title("⚖️ Legal Document RAG Assistant")
st.caption("Ask questions — answers come only from your uploaded documents.")
st.divider()

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander(f"📎 {len(entry['sources'])} source(s)"):
            for i, s in enumerate(entry["sources"], 1):
                st.markdown(f"**[{i}] {s['source']}** `score: {s['score']}`")
                st.caption(s["excerpt"])

query = st.chat_input("Ask a legal question...")
if query:
    if not st.session_state.indexed:
        st.error("Build the index first.")
    elif not groq_key:
        st.error("Enter your Groq API key.")
    else:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(query, st.session_state.index, st.session_state.embedder, st.session_state.chunks, groq_key)
                    st.write(result["answer"])
                    with st.expander(f"📎 {len(result['sources'])} source(s)"):
                        for i, s in enumerate(result["sources"], 1):
                            st.markdown(f"**[{i}] {s['source']}** `score: {s['score']}`")
                            st.caption(s["excerpt"])
                    st.session_state.history.append(result)
                except Exception as e:
                    st.error(f"Error: {e}")
