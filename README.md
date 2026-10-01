# Animal Knowledge RAG Assistant

A portfolio/demo application designed to answer animal-related questions using public/demo documents and cite its sources. Built with a Retrieval-Augmented Generation (RAG) architecture.

## Current Status

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | Project Design | ✅ CEO Approved | System architecture & roadmap defined |
| 2 | GitHub Setup | ✅ CEO Approved | Repository foundation & baseline workflow |
| 3 | Demo Dataset | ✅ CEO Approved | 9 PDFs / 85 physical pages with manifest metadata |
| 4 | Document Ingestion | ✅ CEO Approved | Page-level raw text extraction & manifest validation |
| 5 | Text Processing & Chunking | ✅ CEO Approved | Light text normalization & page-bounded paragraph chunking |
| 6 | Embeddings | 🟡 Code Corrected / Awaiting API Key | 768-dim vector generation via official Google Gen AI SDK (`response.embeddings`) |
| 7 | Vector Storage | 🔲 Planned | FAISS vector index management |
| 8 | Retrieval System | 🔲 Planned | Top-K context retrieval (initially K=4) |
| 9 | RAG Generation | 🔲 Planned | Gemini answer generation with context |
| 10 | Citations & Grounding | 🔲 Planned | Source/page citation resolution |
| 11 | Web Application | 🔲 Planned | Streamlit web interface |
| 12 | Deployment | 🔲 Planned | Streamlit Community Cloud deployment |
| 13 | Portfolio Integration | 🔲 Planned | Documentation & showcase materials |
| 14 | Evaluation & Interview Readiness | 🔲 Planned | Golden QA evaluation & walkthrough prep |

> **Note:** Document ingestion (Stage 4), text processing & chunking (Stage 5), and embeddings module architecture (Stage 6) are fully implemented with **50 automated unit tests passing** (100% offline using SDK response object types and mocks). End-to-end vector storage (FAISS), retrieval, Gemini answering, and Streamlit UI remain planned for future stages.

## Stage 4 — Document Ingestion Overview

Stage 4 implements robust, page-level PDF text extraction and strict dataset validation using [PyMuPDF](https://pymupdf.readthedocs.io/).

### Dataset Corpus
- **9 documents / 85 physical pages**
- Manifest: `data/dataset_manifest.csv`
- Source PDFs: `data/raw/`
- Total raw extracted characters: **500,649**

## Stage 5 — Text Processing & Chunking Overview

Stage 5 implements non-destructive text normalization and deterministic, page-bounded paragraph chunking (`src/processing.py` and `src/chunking.py`).

### Corpus Metrics & Integrity
- **Total Chunks Generated:** **272 chunks** across 9 documents and 85 distinct physical pages.
- **Chunk Sizing:** Target 450 tokens (~1,800 chars), Hard Max 600 tokens (~2,400 chars).
- **Deterministic Chunk IDs:** `{document_id}_p{page_number:03d}_c{chunk_index:03d}`.

## Stage 6 — Embeddings Overview

Stage 6 implements 768-dimensional vector embedding generation using the official `google-genai` SDK and the `gemini-embedding-2` model (`src/embeddings.py`).

### Embedding Specifications & SDK Response Contract
- **SDK**: `google-genai==2.26.0` (`from google import genai`, `from google.genai import types`)
- **Model Identifier**: `gemini-embedding-2`
- **Output Dimensionality**: 768 (`config=types.EmbedContentConfig(output_dimensionality=768)`)
- **SDK Response Contract**: Reads the plural field `response.embeddings`, requires exactly one returned embedding object (`len(response.embeddings) == 1`), and validates `response.embeddings[0].values`. Zero embeddings or multiple embeddings are explicitly rejected.
- **Document Input Format**: `title: {title} | text: {chunk_text}` (preserves original chunk text and metadata without injecting IDs or URLs into semantic input)
- **Reserved Question Format (Stage 8)**: `task: question answering | query: {question}`
- **Request Strategy**: One chunk per request (sequential execution with bounded retries and exponential backoff).
- **Vector Integrity**: Verifies that returned vectors have length 768, contain 100% finite numeric floats, and have a non-zero L2 norm. Returned vectors are preserved as-is without re-normalization.

### Secure Environment Configuration
Live API execution reads `GEMINI_API_KEY` directly from the process environment:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
> **Important:** Creating a `.env` file does not automatically load it. `GEMINI_API_KEY` must be configured in the environment used to launch the command.

If `GEMINI_API_KEY` is not set, the pipeline fails cleanly with:
`ValueError: GEMINI_API_KEY is not set.`

## Running Execution and Tests

### 1. Installation
Install verified dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Automated Offline Test Suite
Execute the full offline unit test suite covering Stages 4, 5, and 6 (requires no network calls or API keys):
```bash
python -m unittest discover tests
```
*Coverage:* **50 automated tests passing** (12 Stage 4 ingestion, 8 Stage 5 processing, 11 Stage 5 chunking, 19 Stage 6 embedding offline tests using SDK response types).

### 3. Run Live Embeddings Pipeline (Requires GEMINI_API_KEY in Process Environment)
To run a 1-chunk smoke test:
```bash
python -m src.embeddings --smoke
```
To run a 3-chunk representative sample test (Factsheet, PLOS, Frontiers):
```bash
python -m src.embeddings --sample
```
To run full-corpus 272-chunk embedding generation:
```bash
python -m src.embeddings
```
*Output Artifact:* Saved atomically to `data/processed/chunk_embeddings.json` (gitignored).

## RAG Pipeline Architecture (Planned Workflow)

```
Text-based PDFs (data/raw/)
  → Stage 4: Page-Level Raw Text Extraction (PyMuPDF) (Completed)
  → Stage 5: Light Text Normalization & Page-Bounded Chunking (Completed)
  → Stage 6: Embeddings (Gemini Embedding 2, 768-dim) (Implemented / Awaiting Key)
  → Stage 7: Vector Storage (FAISS Index) (Planned)
  → Stage 8: Retrieval System (Top-K Context) (Planned)
  → Stage 9: RAG Generation (Gemini LLM) (Planned)
  → Stage 10: Citations & Grounding (Planned)
  → Stage 11: Web Application (Streamlit) (Planned)
```

## Project Structure

```
animal-knowledge-rag/
├── app.py                  # Streamlit application entry point (placeholder)
├── requirements.txt        # Verified project dependencies (pymupdf==1.28.2, google-genai==2.26.0)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # Root documentation
├── src/
│   ├── ingestion.py        # Stage 4: Manifest validation & PDF text extraction
│   ├── processing.py       # Stage 5: Light text normalization
│   ├── chunking.py         # Stage 5: Page-bounded paragraph chunking
│   ├── embeddings.py       # Stage 6: 768-dim Gemini chunk embeddings
│   ├── vector_store.py     # Stage 7: FAISS index management (planned)
│   ├── retrieval.py        # Stage 8: Top-K document retrieval (planned)
│   ├── generation.py       # Stage 9: Gemini answer generation (planned)
│   └── citations.py        # Stage 10: Source/page citation resolution (planned)
├── data/
│   ├── dataset_manifest.csv # Approved dataset manifest (9 PDFs / 85 pages)
│   ├── README.md           # Dataset licensing, attribution, and provenance documentation
│   ├── raw/                # Approved source PDF documents
│   └── processed/          # Intermediate extracted data (gitignored)
├── index/                  # FAISS index files (gitignored)
├── evaluation/             # Evaluation datasets (planned)
└── tests/
    ├── test_ingestion.py   # Stage 4 ingestion test suite
    ├── test_processing.py  # Stage 5 text processing test suite
    ├── test_chunking.py    # Stage 5 chunking test suite
    └── test_embeddings.py  # Stage 6 embedding offline test suite
```

## Data and Licensing Policy

This repository uses strictly public-domain and open-access Creative Commons (CC-BY 4.0) animal research documents. Dataset provenance, licensing evidence, and redistribution terms are detailed in [data/README.md](data/README.md).

## Secrets Policy

- Real API keys and credentials must never be committed to this repository.
- The `.env.example` file provides the template for required environment variables.

## License

See dataset details and licensing notes in `data/README.md`.
