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
| 6 | Embeddings | ✅ CEO Approved | 768-dim vector generation via official Google Gen AI SDK (272 chunks embedded) |
| 7 | Vector Storage | ✅ CEO Approved | FAISS IndexFlatIP 768-dim vector index & metadata mapping |
| 8 | Retrieval System | ✅ CEO Approved | Semantic Top-K context retrieval (K=4) with bounded retries & compatibility checks |
| 9 | RAG Generation | ✅ CEO Approved | Grounded answer generation via gemini-3.8-flash & controlled C1..CK source IDs |
| 10 | Citations & Grounding | 🟡 Needs CEO Review | Source/page citation resolution & fail-closed validation |
| 11 | Web Application | 🔲 Planned | Streamlit web interface |
| 12 | Deployment | 🔲 Planned | Streamlit Community Cloud deployment |
| 13 | Portfolio Integration | 🔲 Planned | Documentation & showcase materials |
| 14 | Evaluation & Interview Readiness | 🔲 Planned | Golden QA evaluation & walkthrough prep |

> **Note:** Document ingestion (Stage 4), text processing & chunking (Stage 5), embeddings (Stage 6), vector storage (Stage 7), retrieval system (Stage 8), grounded RAG generation (Stage 9), and citation resolution (Stage 10) are fully implemented and verified with **164 automated unit tests passing**. Streamlit UI (Stage 11) remains planned for future stages.


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

### Live Validation & Execution Results
- **Full Corpus Execution:** 272 / 272 chunks embedded successfully.
- **Model Identifier:** `gemini-embedding-2`
- **Output Dimensionality:** 768 dimensions per vector.
- **Failed Chunks:** 0
- **Missing, Extra, or Duplicate IDs:** 0
- **Text & Metadata Preservation:** Exact preservation of original text, titles, document IDs, page numbers, and filenames.
- **Generated Artifact:** Saved atomically to `data/processed/chunk_embeddings.json` (4.46 MB, validated, and kept gitignored / outside Git).
- **Automated Test Suite:** 50 automated tests passing, as recorded in the accepted evidence.

#### Measured L2 Norm Statistics
- **Minimum L2 Norm:** `0.999999105449`
- **Mean L2 Norm:** `0.999999996118`
- **Maximum L2 Norm:** `1.000000603547`

#### Runtime & Rate Limit Pacing Performance
- **Configured Inter-Request Pacing:** 4.2 seconds delay between sequential requests (14.28 RPM) to adhere strictly to the Gemini API Free Tier 15 RPM rate limit.
- **Theoretical Pacing-Only Delay:** 271 × 4.2 seconds = 1,138.2 seconds, approximately 19 minutes.
- **Observed End-to-End Runtime:** Approximately 24 minutes, based on the recorded execution evidence.

## Stage 7 — Vector Storage Overview

Stage 7 implements FAISS vector index construction, float32 matrix normalization, integer-ID-to-metadata mapping, and persistent storage (`src/vector_store.py`).

### Vector Storage Specifications & Architecture
- **Index Class**: `faiss.IndexFlatIP(768)` (Flat inner-product index using `METRIC_INNER_PRODUCT`).
- **Matrix Contract**: Independent 2D `float32` C-contiguous NumPy matrix of shape `(272, 768)` converted directly from the Stage 6 embeddings artifact.
- **Defensive L2 Normalization**: Row-wise L2-normalized using `faiss.normalize_L2(matrix)` prior to index insertion. Post-normalization row norms are verified to be ~1.0 (`Min=1.000000, Max=1.000000, Mean=1.000000`).
- **Cosine-via-Inner-Product Contract**: Document vectors are normalized before storage. When Stage 8 normalizes query vectors, inner product matches cosine similarity.
- **Integer ID Mapping**: FAISS IDs `0` through `271` mapped directly to chunk metadata dictionaries. Full 768-element vector array is excluded from metadata JSON to prevent redundant storage.
- **Offline Guarantee**: Requires **zero Gemini API calls** and operates without `GEMINI_API_KEY`.
- **Reconstruction Verification**: 100% accuracy (`272/272` vectors reconstructed via `index.reconstruct(i)` matched stored float32 matrix within numerical comparison tolerance `rtol=1e-5, atol=1e-5`).

### Persistent Index Artifacts (`index/`)
Written atomically via temporary `.tmp` files to prevent partially updated states:
- `index/faiss.index`: Native FAISS index binary file.
- `index/chunk_metadata.json`: Deterministic JSON metadata array.
- `index/index_manifest.json`: Index parameters, file SHA-256 checksums, and ordered chunk IDs fingerprint (`ordered_chunk_ids_sha256`).
- *All `index/*` files are ignored by Git via `.gitignore` (except `.gitkeep`).*

## Stage 8 — Retrieval System Overview

Stage 8 implements semantic Top-K document chunk retrieval using Gemini query embeddings and local FAISS vector search (`src/retrieval.py`).

### Retrieval Specifications & Architecture
- **Query Format**: `task: question answering | query: {question}` (preserves original user question string without lowercasing or transforming).
- **Query Model & Dimension**: `gemini-embedding-2` producing 768-dimensional float vectors via official `google-genai` SDK (`client.models.embed_content`).
- **Bounded Query-Embedding Retries**: Configurable `max_retries=3`, `retry_delay=1.0` (allowing up to 4 total attempts). Implements capped exponential backoff sleeping only between attempts. Non-transient errors (401, 403, 400) fail immediately. Transient errors (503, 429) retry up to the retry limit. Exhausted retries raise an explicit failure.
- **Query/Index Compatibility Enforcement**: Verifies query `model_name` (`gemini-embedding-2`) and `dimension` (`768`) match `manifest["embedding_model"]`, `manifest["dimension"]`, and FAISS index dimension `index.d` **prior** to client initialization or making live API calls.
- **Query Matrix Normalization**: Converted to 2D `float32` C-contiguous array `(1, 768)` and row-wise L2-normalized via `faiss.normalize_L2(query_matrix)` prior to search.
- **Top-K Search Strategy**: Default `DEFAULT_TOP_K = 4`. Performs `scores, ids = index.search(query_matrix, top_k)` against the Stage 7 `IndexFlatIP` store. Inner product scores represent exact cosine similarity.
- **Prerequisite Validation Order**: Question non-emptiness, index store availability, `top_k` bounds (`1 <= top_k <= 272`), and query/index model & dimension compatibility are validated **before** invoking the Gemini API endpoint, protecting API quota from invalid calls.
- **Full-Text Result Contract**: Each returned result entry includes `rank` (1..K), `faiss_id`, `similarity_score`, `chunk_id`, `document_id`, `filename`, `page_number`, `chunk_index`, full un-truncated `text` (preserved for Stage 9 RAG generation), `title`, `publisher`, `source_url`, `estimated_token_count`, `embedding_model`, and `embedding_dimension`. Vector arrays are excluded.
- **Sanity Set Scope & Retrieval Results**: Evaluated on an 8-question, hand-selected cross-corpus sanity set covering 8 target source documents across the approved 9-document corpus. The Monarch Butterfly factsheet (`monarch_butterfly_factsheet.pdf`) was not a dedicated expected source in this set. Achieving 8/8 describes expected-source presence for these hand-selected questions; it is not a general retrieval-accuracy measurement. Source-file presence alone does not establish that every returned chunk supports an answer.
- **Baseline Preservation**: Reranking remains deferred. The dense Top-K retrieval baseline is preserved as-is.

## Stage 9 — RAG Generation Overview

Stage 9 implements grounded answer generation using the Gemini 3.8 Flash model, trusted system instructions, controlled `C1..CK` source identifiers, and Stage 8 retrieval context (`src/generation.py`).

### Generation Specifications & Architecture
- **Generation Model**: `gemini-3.8-flash` via official `google-genai` SDK (`client.models.generate_content`).
- **Thinking Level**: Configured to `low` (`types.ThinkingConfig(thinking_level="low")`) to reduce latency and reasoning overhead.
- **Generation Hyperparameters**: Temperature, `top_p`, and `top_k` remain un-overridden (using Gemini 3.x default parameters).
- **Prohibition of External Knowledge & Tools**: No external tools (`google_search`, `url_context`, `code_execution`, `file_search`) are enabled. Generation is strictly grounded in local retrieved context.
- **Trusted System Instruction**: Explicitly treats source context as untrusted data blocks (`[C1]...[/C1]`), prohibiting prompt-injection overrides and constraining answers to retrieved facts.
- **Controlled Source Identifiers**: Maps Stage 8 retrieval rank order deterministically to controlled generation source IDs (`C1`, `C2`, `C3`, `C4`).
- **Exact Grounded Fallback Sentence**: If retrieved context is insufficient, the system emits:
  `"I don't have enough information in the provided sources to answer that question."`
- **Citation Rendering Boundary**: Raw `[C1]` markers and `source_map` records are returned as-is. Final citation parsing, validation, and filename/page rendering remain deferred to Stage 10 (`src/citations.py`).

## Stage 10 — Citations & Grounding Overview

Stage 10 implements a deterministic citation parsing, fail-closed validation, trusted metadata lookup, and user-facing filename/page rendering pipeline (`src/citations.py`).

### Citation Specifications & Architecture
- **Citation Authority**: Application code (Gemini output is NOT authoritative citation metadata).
- **Controlled Marker Grammar**: Validates `[C1]`, `[C1, C2]`, `[C1, C2, C4]` markers. Rejects `[C0]`, `[C01]`, `[c1]`, `[C1; C2]`, `[C1 C2]`, `[C1,]`, and duplicate markers `[C1, C1]`.
- **Fail-Closed Policy**: Rejects unknown source IDs (e.g. `[C99]`), malformed marker syntax, or supported non-fallback answers missing citations.
- **Trusted Metadata Lookup**: Filenames and 1-based physical page numbers are read exclusively from Stage 9 `source_map` records.
- **User-Facing Inline Rendering**:
  - Single page: `(bald_eagle_lead_exposure.pdf, p. 1)`
  - Multiple pages same document: `(african_elephant_reintegration.pdf, pp. 2, 3)`
  - Multiple documents: `(file_a.pdf, p. 1; file_b.pdf, pp. 2, 4)`
  - Identical file/page references within the same group collapse to a single reference.
- **Exact Fallback Match**: The exact fallback (`"I don't have enough information in the provided sources to answer that question."`) contains 0 citations (`citation_group_count = 0`, `is_fallback = True`).
- **Offline Processing Guarantee**: Resolving citations (`resolve_rag_citations`) operates 100% locally with 0 API calls, 0 network requests, and no API key required.

## Running Execution and Tests

### 1. Installation
Install verified dependencies:
```bash
pip install -r requirements.txt
```
*Pinned Dependencies:* `pymupdf==1.28.2`, `google-genai==2.26.0`, `numpy==2.4.3`, `faiss-cpu==1.15.1`.

### 2. Run Automated Offline Test Suite
Execute the full offline unit test suite covering Stages 4, 5, 6, 7, 8, 9, and 10 (requires no network calls or API keys):
```bash
python -m unittest discover tests
```
*Coverage:* **164 automated tests passing** (12 Stage 4 ingestion, 8 Stage 5 processing, 11 Stage 5 chunking, 19 Stage 6 embedding, 22 Stage 7 vector store, 24 Stage 8 retrieval, 20 Stage 9 generation, 48 Stage 10 citation offline tests).

### 3. Run Stage 8 Retrieval CLI (Requires GEMINI_API_KEY)
To execute live Top-K semantic retrieval for a user question:
```bash
python -m src.retrieval --query "How long do bald eagle eggs take to hatch?" --top-k 4
```

### 4. Run Stage 9 Grounded RAG Generation CLI (Requires GEMINI_API_KEY)
To execute end-to-end retrieval and grounded answer generation:
```bash
python -m src.generation --query "How does lead ammunition expose bald eagles to lead?" --top-k 4
```

### 5. Run Stage 10 Citations & Grounding CLI (Requires GEMINI_API_KEY)
To execute end-to-end RAG with filename and page citation rendering:
```bash
python -m src.citations --query "How does lead ammunition expose bald eagles to lead?" --top-k 4
```
*Requirement:* Existing Stage 7 index artifacts (`index/`) and `GEMINI_API_KEY` configured in the process environment. `.env.example` is a template only; application code does not automatically load `.env`.

## RAG Pipeline Architecture

```
Text-based PDFs (data/raw/)
  → Stage 4: Page-Level Raw Text Extraction (PyMuPDF) (Completed)
  → Stage 5: Light Text Normalization & Page-Bounded Chunking (Completed)
  → Stage 6: Embeddings (Gemini Embedding 2, 768-dim) (Completed)
  → Stage 7: Vector Storage (FAISS IndexFlatIP) (Completed)
  → Stage 8: Retrieval System (Top-K Context) (Completed)
  → Stage 9: Grounded RAG Generation (Gemini 3.8 Flash) (Completed)
  → Stage 10: Citations & Grounding (Deterministic Resolution) (Awaiting CEO Review)
  → Stage 11: Web Application (Streamlit) (Planned)
```

## Project Structure

```
animal-knowledge-rag/
├── app.py                  # Streamlit application entry point (placeholder)
├── requirements.txt        # Verified project dependencies (pymupdf, google-genai, numpy, faiss-cpu)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── README.md               # Root documentation
├── src/
│   ├── ingestion.py        # Stage 4: Manifest validation & PDF text extraction
│   ├── processing.py       # Stage 5: Light text normalization
│   ├── chunking.py         # Stage 5: Page-bounded paragraph chunking
│   ├── embeddings.py       # Stage 6: 768-dim Gemini chunk embeddings
│   ├── vector_store.py     # Stage 7: FAISS IndexFlatIP index & metadata storage
│   ├── retrieval.py        # Stage 8: Semantic Top-K document chunk retrieval
│   ├── generation.py       # Stage 9: Gemini grounded RAG answer generation
│   └── citations.py        # Stage 10: Source/page citation resolution
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
    ├── test_embeddings.py  # Stage 6 embedding offline test suite
    ├── test_vector_store.py# Stage 7 vector storage test suite
    ├── test_retrieval.py   # Stage 8 retrieval test suite
    ├── test_generation.py  # Stage 9 generation offline test suite
    └── test_citations.py   # Stage 10 citation offline test suite
```


## Data and Licensing Policy

This repository uses strictly public-domain and open-access Creative Commons (CC-BY 4.0) animal research documents. Dataset provenance, licensing evidence, and redistribution terms are detailed in [data/README.md](data/README.md).

## Secrets Policy

- Real API keys and credentials must never be committed to this repository.
- The `.env.example` file provides the template for required environment variables.

## License

See dataset details and licensing notes in `data/README.md`.
