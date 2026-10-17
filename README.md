# DSA Tutor RAG

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Author: Anuj Mhatre](https://img.shields.io/badge/Author-Anuj%20Mhatre-purple)

## Why We Built This

As a second-year BTech student, I wanted a smarter way to study DSA. Instead of Googling the same algorithms repeatedly, I built a RAG system that answers questions **using my own notes and C++ implementations**. No more context-switching — just ask a question and get grounded answers with citations.

## What It Does

A **Retrieval-Augmented Generation (RAG)** tutor that ingests your DSA notes (C++, Markdown, text), builds a hybrid search index, and answers questions using an LLM — all grounded in your own content with citations.

- **Hybrid search**: Vector similarity + BM25 + Reciprocal Rank Fusion
- **Cross-encoder reranking** for precision
- **Streamlit chat UI** with source citations
- **CLI mode** for quick queries

## How It Was Built

| Layer | Tech |
|-------|------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Keyword Search | `rank_bm25` (BM25Okapi) |
| Reranking | Cross-encoder (ms-marco-MiniLM-L6-v2) |
| LLM | OpenAI-compatible API (configurable) |
| UI | Streamlit |
| Chunking | Custom function-aware splitter for C++ code |

**Architecture**: `ingest.py` → chunk → embed → store in ChromaDB. At query time, `hybrid_search()` runs vector + BM25 in parallel, fuses with RRF, reranks, then feeds top-k to the LLM.

## Quick Start

```bash
# Clone & install
git clone https://github.com/a18-n03/dsa-tutor-rag.git
cd dsa-tutor-rag
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your ZEN_API_KEY

# Ingest your notes (put them in data/notes/)
python ingest.py

# Run the chat UI
streamlit run app.py

# Or use CLI
python tutor.py "How does Kruskal work?"
```

## Project Structure

```
dsa-tutor-rag/
├── app.py                  # Streamlit chat UI
├── tutor.py                # CLI interface
├── ingest.py               # Index builder
├── src/
│   ├── chunking.py         # Code-aware text chunking
│   ├── retrieval.py        # Hybrid search + reranking
│   ├── llm.py              # LLM client wrapper
│   └── eval.py             # RAG evaluation suite
├── data/notes/             # Your DSA notes go here
├── test_retrieval.py       # Retrieval test script
├── test_hybrid.py          # Hybrid search benchmark
└── test_e2e.py             # End-to-end LLM test
```

## Test Results

```
Test: Variable 'i' undefined in sidebar rendering loop (app.py:104)
Status: FIXED — Changed `for h in hits:` to `for i, h in enumerate(hits):`

All source files parse correctly. No syntax errors found.
```
