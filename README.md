# Animal Knowledge RAG Assistant

A portfolio/demo application designed to answer animal-related questions using public/demo documents and cite its sources. Built with a Retrieval-Augmented Generation (RAG) architecture.

## Current Status

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | Project Design | ✅ CEO Approved | System architecture & roadmap defined |
| 2 | GitHub Setup | ✅ CEO Approved | Repository foundation & baseline workflow |
| 3 | Demo Dataset | ✅ CEO Approved | 9 PDFs / 85 physical pages with manifest metadata |
| 4 | Document Ingestion | 🟡 Awaiting CEO Review | Page-level raw text extraction & manifest validation |
| 5 | Text Processing & Chunking | 🔲 Planned | Text normalization & page/paragraph-aware chunking |
| 6 | Embeddings | 🔲 Planned | Gemini Embedding 2 (768-dimensional output) |
| 7 | Vector Storage | 🔲 Planned | FAISS vector index management |
| 8 | Retrieval System | 🔲 Planned | Top-K context retrieval (initially K=4) |
| 9 | RAG Generation | 🔲 Planned | Gemini answer generation with context |
| 10 | Citations & Grounding | 🔲 Planned | Source/page citation resolution |
| 11 | Web Application | 🔲 Planned | Streamlit web interface |
| 12 | Deployment | 🔲 Planned | Streamlit Community Cloud deployment |
| 13 | Portfolio Integration | 🔲 Planned | Documentation & showcase materials |
| 14 | Evaluation & Interview Readiness | 🔲 Planned | Golden QA evaluation & walkthrough prep |

> **Note:** Document ingestion and automated testing for Stage 4 are fully implemented. End-to-end vector retrieval, Gemini answering, and Streamlit UI remain planned for future stages.

## Stage 4 — Document Ingestion Overview

Stage 4 implements robust, page-level PDF text extraction and strict dataset validation using [PyMuPDF](https://pymupdf.readthedocs.io/).

### Dataset Corpus

The approved dataset contains **9 documents / 85 physical pages**:
- Metadata manifest: `data/dataset_manifest.csv`
- Raw PDF files: `data/raw/`

### Ingestion Contract (Page Record Structure)

Each page record is returned as a standard dictionary containing the following keys:
- `document_id`: Unique manifest document identifier (e.g., `doc_bald_eagle`)
- `filename`: PDF filename in `data/raw/` (e.g., `bald_eagle_factsheet.pdf`)
- `page_number`: 1-based physical page number in the PDF (1, 2, ...)
- `text`: Complete, uncleaned raw text extracted from the page
- `title`: Publication title from manifest metadata
- `publisher`: Publisher name from manifest metadata
- `source_url`: Canonical URL source from manifest metadata

### Validation and Error Handling

The ingestion pipeline (`src/ingestion.py`) performs strict validation before and during extraction:
- **Manifest Integrity**: Verifies required CSV headers exist and required metadata values are non-blank.
- **Uniqueness**: Ensures all `document_id` and `filename` entries in the manifest are strictly unique.
- **File & Path Safety**: Confirms all listed PDFs exist in `data/raw/` and resolve within the raw data directory.
- **Unmanifested Files**: Rejects unexpected unmanifested `.pdf` files in `data/raw/` (ignores `.gitkeep`).
- **Page Count Agreement**: Verifies actual PDF page counts match recorded manifest counts.
- **Empty-Page Policy**: Explicitly fails with a `ValueError` if a physical page contains empty or whitespace-only text.

### Extraction Strategy & Limitations

Text is extracted using `page.get_text("text", sort=True)`, which orders text blocks top-to-bottom and left-to-right to improve reading order on multi-column layouts (e.g., PLOS ONE and Frontiers scientific articles).

**Boundary Notice**: Stage 4 returns raw extracted text without modification. Normalization (whitespace cleaning, header/footer removal, dehyphenation), paragraph reconstruction, and chunking are explicitly deferred to Stage 5+.

## Running Ingestion and Tests

### 1. Installation
Install the tested dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Corpus Ingestion
Run full-corpus ingestion via the module entrypoint:
```bash
python -m src.ingestion
```
*Output Summary:* Successfully ingests 9 documents yielding 85 page records across **500,649** total extracted characters (the exact sum of `len(record["text"])` across all 85 raw page records produced by the current ingestion configuration, not a token count).

### 3. Run Automated Tests
Execute the focused `unittest` test suite:
```bash
python -m unittest discover tests
```
*Coverage:* 12 test cases verifying manifest loading, metadata preservation, file existence, duplicate detection, unlisted PDF rejection, page numbering contiguity, exact 9-doc / 85-page corpus yields, and controlled failure handling.

## RAG Pipeline Architecture (Planned Workflow)

```
Text-based PDFs (data/raw/)
  → Stage 4: Page-Level Raw Text Extraction (PyMuPDF)
  → Stage 5: Text Processing & Chunking (Planned)
  → Stage 6: Embeddings (Planned)
  → Stage 7: Vector Storage (Planned)
  → Stage 8: Retrieval System (Planned)
  → Stage 9: RAG Generation (Planned)
  → Stage 10: Citations & Grounding (Planned)
  → Stage 11: Web Application (Planned)
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
│   ├── processing.py       # Stage 5: Text processing & chunking (planned)
│   ├── chunking.py         # Stage 5: Page/paragraph-aware chunking (planned)
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
    └── test_ingestion.py   # Stage 4 ingestion test suite
```

## Data and Licensing Policy

This repository uses strictly public-domain and open-access Creative Commons (CC-BY 4.0) animal research documents. Dataset provenance, licensing evidence, and redistribution terms are detailed in [data/README.md](data/README.md).

## Secrets Policy

- Real API keys and credentials must never be committed to this repository.
- The `.env.example` file provides the template for required environment variables.

## License

See dataset details and licensing notes in `data/README.md`.
