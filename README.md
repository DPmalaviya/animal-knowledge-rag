# Animal Knowledge RAG Assistant

A portfolio/demo application planned to answer animal-related questions using public/demo documents and cite its sources. Built with a Retrieval-Augmented Generation (RAG) architecture.

## Current Status

| Stage | Status |
|-------|--------|
| Stage 1 — Design | ✅ Approved |
| Stage 2 — Repository foundation | ✅ Prepared |
| Stage 3+ — Implementation | 🔲 Not started |

> **Note:** There is no runnable application or deployed demo yet. This repository currently contains only the project scaffolding and placeholder modules.

## Planned Workflow

```
Text-based PDFs
  → Extraction (PyMuPDF)
  → Light normalization
  → Page/paragraph-aware chunking
  → Embeddings (Gemini Embedding 2, 768 dimensions)
  → FAISS retrieval (Top-K, initially K=4)
  → Answer generation (Gemini)
  → Filename/page citations
```

## Planned Architecture

| Component | Planned Technology |
|---|---|
| Language | Python |
| PDF extraction | PyMuPDF (text-based extraction) |
| Embeddings | Gemini Embedding 2 (768-dimensional output) |
| Vector store | FAISS with normalized vectors and inner-product search (cosine similarity) |
| Retrieval | Top-K, initially K=4 |
| Orchestration | Plain Python |
| Generation | Gemini |
| Citations | Controlled source/chunk IDs resolved to filename and page |
| UI | Streamlit |
| Deployment target | Streamlit Community Cloud |
| Secrets management | Environment variables / deployment secrets |
| Evaluation | Small manually verified dataset (planned for a later stage) |

All technologies listed above are planned choices approved in Stage 1. Integration has not been implemented or validated.

## Project Structure

```
animal-knowledge-rag/
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies (placeholder)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── src/
│   ├── ingestion.py        # PDF text extraction
│   ├── processing.py       # Text normalization
│   ├── chunking.py         # Page/paragraph-aware chunking
│   ├── embeddings.py       # Gemini embedding generation
│   ├── vector_store.py     # FAISS index management
│   ├── retrieval.py        # Top-K document retrieval
│   ├── generation.py       # Gemini answer generation
│   └── citations.py        # Source/page citation resolution
├── data/
│   ├── raw/                # Source PDF documents
│   └── processed/          # Extracted/chunked data (generated, not committed)
├── index/                  # FAISS index files (generated, not committed)
├── evaluation/             # Evaluation datasets and results
└── tests/                  # Test files
```

### Folder Responsibilities

- **`src/`** — Application source modules covering the full RAG pipeline.
- **`data/raw/`** — Source PDF documents. Public/demo PDFs may be committed in later stages.
- **`data/processed/`** — Generated intermediate data (gitignored; only `.gitkeep` is tracked).
- **`index/`** — Generated FAISS index files (gitignored; only `.gitkeep` is tracked).
- **`evaluation/`** — Future evaluation datasets and results.
- **`tests/`** — Future test files.

## Data Policy

This project will use only **public or demo animal information documents**. No private, personal, or confidential documents should be added to the repository or used with the application.

## Secrets Policy

- The application requires a `GEMINI_API_KEY` environment variable.
- For local development, configure it in a `.env` file (which is gitignored).
- For deployment, configure it through Streamlit Community Cloud's secrets management.
- **Real API keys and credentials must never be committed to this repository.**
- The `.env.example` file contains only the variable name with no value.

## License

*To be determined.*
