import os, re, pickle
from pathlib import Path
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL  = "llama3-8b-8192"
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 100
TOP_K = 5

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_text_from_txt(path):
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

def build_chunks(documents):
    all_chunks = []
    for doc in documents:
        text = clean_text(doc["text"])
        start = 0
        while start < len(text):
            chunk = text[start:start + CHUNK_SIZE]
            if chunk.strip():
                all_chunks.append({"text": chunk, "source": doc["source"]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks

def build_index(chunks, save=False):
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embedder, chunks

def retrieve(query, index, embedder, chunks):
    q = embedder.encode([query], convert_to_numpy=True).astype("float32")
    q = np.ascontiguousarray(q)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, TOP_K)
    return [{**chunks[i], "score": round(float(s), 3)} for s, i in zip(scores[0], indices[0]) if i < len(chunks)]

def ask(query, index, embedder, chunks, api_key):
    retrieved = retrieve(query, index, embedder, chunks)
    context = "\n\n---\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in retrieved)
    prompt = f"""You are a legal document assistant. Answer ONLY based on the documents below.
If the answer is not in the documents, say so clearly.

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "sources": [{"source": r["source"], "score": r["score"], "excerpt": r["text"][:200] + "..."} for r in retrieved]
    }
