# Animal Knowledge RAG Assistant

A portfolio/demo application designed to answer animal-related questions using public/demo documents and cite its sources. Built with a Retrieval-Augmented Generation (RAG) architecture.

## Current Status

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | Project Design | ✅ CEO Approved | System architecture & roadmap defined |
| 2 | GitHub Setup | ✅ CEO Approved | Repository foundation & baseline workflow |
| 3 | Demo Dataset | ✅ CEO Approved | 9 PDFs / 85 physical pages with manifest metadata |
| 4 | Document Ingestion | ✅ CEO Approved | Page-level raw text extraction & manifest validation |
| 5 | Text Processing & Chunking | 🟡 Awaiting CEO Review | Light text normalization & page-bounded paragraph chunking |
| 6 | Embeddings | 🔲 Planned | Gemini Embedding 2 (768-dimensional output) |
| 7 | Vector Storage | 🔲 Planned | FAISS vector index management |
| 8 | Retrieval System | 🔲 Planned | Top-K context retrieval (initially K=4) |
| 9 | RAG Generation | 🔲 Planned | Gemini answer generation with context |
| 10 | Citations & Grounding | 🔲 Planned | Source/page citation resolution |
| 11 | Web Application | 🔲 Planned | Streamlit web interface |
| 12 | Deployment | 🔲 Planned | Streamlit Community Cloud deployment |
| 13 | Portfolio Integration | 🔲 Planned | Documentation & showcase materials |
| 14 | Evaluation & Interview Readiness | 🔲 Planned | Golden QA evaluation & walkthrough prep |

> **Note:** Document ingestion (Stage 4) and text processing & chunking (Stage 5) are fully implemented with **31 automated tests passing**. End-to-end vector embeddings, retrieval, Gemini answering, and Streamlit UI remain planned for future stages.

## Stage 4 — Document Ingestion Overview

Stage 4 implements robust, page-level PDF text extraction and strict dataset validation using [PyMuPDF](https://pymupdf.readthedocs.io/).

### Dataset Corpus
- **9 documents / 85 physical pages**
- Manifest: `data/dataset_manifest.csv`
- Source PDFs: `data/raw/`
- Total raw extracted characters: **500,649**

## Stage 5 — Text Processing & Chunking Overview

Stage 5 implements non-destructive text normalization and deterministic, page-bounded paragraph chunking (`src/processing.py` and `src/chunking.py`).

### Processing & Normalization Rules (`src/processing.py`)
- **Newline Normalization**: Converts `\r\n` and `\r` to standard `\n`.
- **Horizontal Space Collapse**: Collapses multiple spaces and tabs into a single space per line (`re.sub(r"[ \t]+", " ", line).strip()`).
- **Paragraph Preservation**: Retains double newlines (`\n\n`) to preserve paragraph boundaries while collapsing 3+ consecutive newlines (`\n{3,}` -> `\n\n`).
- **Punctuation & Vocabulary Preservation**: Retains headings, punctuation, capitalization, scientific nomenclature (e.g. *Chelonia mydas*), numbers, and legitimate hyphenated terms ("scent-detection", "non-lead"). Automatic dehyphenation across line wraps is explicitly deferred to avoid misjoining compound terms.
- **Header/Footer Exclusions**: Avoids destructive text deletion. Raw text block integrity is maintained across all 85 source pages.

### Page-Bounded Chunking Strategy (`src/chunking.py`)
- **Page Boundary Contract**: Every chunk belongs strictly to **exactly one physical PDF page** (`page_number`). No text or overlap crosses page boundaries.
- **Deterministic Token Estimation**: `estimate_tokens(text) = max(1, len(text) // 4)` (~4 characters per token).
- **Target Size**: 450 estimated tokens (~1,800 characters).
- **Hard Maximum**: 600 estimated tokens (~2,400 characters). Zero chunks exceed this maximum.
- **Bounded Overlap & Small-Tail Merging**: Up to 75 tokens (~300 characters) of trailing sentences from the preceding chunk on the *same page* are prepended to the subsequent chunk. Small trailing chunks (<100 tokens) are merged back into preceding chunks on the same page when total size <= 600 tokens; duplicate overlap is explicitly prevented during merges by concatenating only the trailing chunk's new source content.
- **Oversized Paragraph/Sentence Fallback**: Paragraphs exceeding target sizes are split into sentences, and oversized sentences are split into word groups. Uninterrupted sequences (e.g. long URLs) are sliced at character thresholds to guarantee loop termination and size compliance.
- **Deterministic Chunk IDs**: `{document_id}_p{page_number:03d}_c{chunk_index:03d}` (1-based chunk index per page).

### Chunk Record Structure
Each generated chunk dictionary contains:
- `chunk_id`: Deterministic unique identifier (e.g. `doc_bald_eagle_p001_c001`)
- `document_id`: Source document identifier
- `filename`: Source PDF filename
- `page_number`: 1-based physical page number
- `chunk_index`: 1-based chunk index on this page
- `text`: Processed chunk text block
- `title`: Publication title from manifest
- `publisher`: Publisher from manifest
- `source_url`: Canonical URL source from manifest
- `estimated_token_count`: Measured token estimate

## Running Execution and Tests

### 1. Installation
Install verified dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Corpus Ingestion (Stage 4)
```bash
python -m src.ingestion
```
*Output Summary:* Successfully ingests 9 documents yielding 85 page records across 500,649 raw extracted characters.

### 3. Run Processing & Chunking Pipeline (Stage 5)
```bash
python -m src.chunking
```
*Measured Metrics Summary:*
- **Input Corpus:** 9 documents / 85 physical pages / 500,649 raw chars.
- **Processed Page Text:** 373,278 characters (whitespace/padding collapsed without text loss).
- **Total Chunks Generated:** **272 chunks** across 9 documents and 85 distinct source pages.
- **Size Metrics:** Minimum: 62 tokens | Maximum: 518 tokens | Mean: 385.2 tokens | Median: 422.0 tokens.
- **Integrity:** 0 empty chunks, 0 duplicate IDs, 0 hard-maximum violations (>600 tokens).

### 4. Run Automated Test Suite
Execute the full `unittest` test suite covering Stages 4 and 5:
```bash
python -m unittest discover tests
```
*Coverage:* **31 automated tests passing** verifying manifest loading, metadata preservation, unlisted file detection, 1-based page contiguity, text normalization, paragraph preservation, page-bounded chunking, overlap rules, small-tail merge overlap deduplication, fallback splitting, and deterministic chunk ID uniqueness.

## RAG Pipeline Architecture (Planned Workflow)

```
Text-based PDFs (data/raw/)
  → Stage 4: Page-Level Raw Text Extraction (PyMuPDF)
  → Stage 5: Light Text Normalization & Page-Bounded Chunking (Completed)
  → Stage 6: Embeddings (Gemini Embedding 2, 768-dim) (Planned)
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
├── requirements.txt        # Verified project dependencies (pymupdf==1.28.2)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # Root documentation
├── src/
│   ├── ingestion.py        # Stage 4: Manifest validation & PDF text extraction
│   ├── processing.py       # Stage 5: Light text normalization
│   ├── chunking.py         # Stage 5: Page-bounded paragraph chunking
│   ├── embeddings.py       # Stage 6: Gemini embedding generation (planned)
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
    └── test_chunking.py    # Stage 5 chunking test suite
```

## Data and Licensing Policy

This repository uses strictly public-domain and open-access Creative Commons (CC-BY 4.0) animal research documents. Dataset provenance, licensing evidence, and redistribution terms are detailed in [data/README.md](data/README.md).

## Secrets Policy

- Real API keys and credentials must never be committed to this repository.
- The `.env.example` file provides the template for required environment variables.

## License

See dataset details and licensing notes in `data/README.md`.
