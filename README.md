Legal Document RAG Assistant

Ask questions about legal documents in plain English — powered by **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and **Groq LLaMA 3**.


## What It Does

Upload any legal PDF (contract, NDA, court order, terms of service) and ask questions about it in natural language. The system retrieves the most relevant sections from your document and grounds every answer strictly in that content — with source citations shown for every response.


## Features

- Upload multiple **PDF or TXT** legal documents
- **Real-time document indexing** with chunk overlap to preserve context
- **Natural language Q&A** grounded strictly in document content
- **Source attribution** — every answer shows which document section it came from
- **Chat history** maintained within the session
- **Hallucination reduction** via low temperature (0.2) LLM settings
- Clean **Streamlit web interface** — no setup needed beyond API key


## Tech Stack

- Python 3.10+
- Groq API (LLaMA 3 8B) — free tier
- FAISS (CPU) — vector similarity search
- Sentence Transformers (all-MiniLM-L6-v2) — local embeddings
- PyMuPDF — PDF text extraction
- Streamlit — web UI


## How It Works

```
User Query
    │
    ▼
[Sentence Transformers]     ← converts query to vector (runs locally)
    │
    ▼
[FAISS Vector Store]        ← finds top-5 most relevant document chunks
    │
    ▼
[Prompt Builder]            ← injects retrieved chunks as context
    │
    ▼
[Groq API — LLaMA 3 8B]    ← generates grounded answer
    │
    ▼
Answer + Source Citations
```

**Why RAG over fine-tuning?**
RAG retrieves facts at query time — no retraining needed when documents change. For legal documents that update frequently, RAG is faster, cheaper, and more reliable than fine-tuning.

---

## Files Included

```
legal-rag-assistant/
|-- rag_engine.py          ← core RAG pipeline (ingest, chunk, embed, retrieve, generate)
|-- streamlit_app.py       ← Streamlit web UI
|-- app.py                 ← CLI interface
|-- requirements.txt       ← all dependencies
|-- README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — no credit card needed.

### 3. Run the app
```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. Paste your **Groq API key** in the sidebar
2. Click **Upload** and add your legal PDF or TXT file
3. Click **Build Index**
4. Type your question in the chat box

### Example Questions

| Document Type | Sample Question |
|---|---|
| NDA | "What information is considered confidential?" |
| Employment Contract | "What is the notice period for resignation?" |
| Terms of Service | "What liability does the company disclaim?" |
| Court Judgment | "What was the final ruling and on what grounds?" |
| Rent Agreement | "What are the conditions for early termination?" |
