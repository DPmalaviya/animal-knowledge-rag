# Case Study: Animal Knowledge RAG Assistant

## Executive Summary

The **Animal Knowledge RAG Assistant** is a portfolio-scale Retrieval-Augmented Generation (RAG) system built to provide factually grounded answers to animal biology, ecology, and conservation questions. Operating over a curated 9-document corpus combining government factsheets and peer-reviewed open-access research (85 physical pages / 272 chunks), the application couples a 768-dimensional dense vector store with Google Gemini 3.8 Flash generation and a deterministic Stage 10 citation parser.

## Problem Statement & Constraints

Large Language Models (LLMs) can generate ungrounded statements or fabricate facts when answering domain-specific scientific queries. In biological and ecological research, unverified claims can misrepresent species conservation statuses, chemical toxicity mechanisms, or physiological indicators.

To build a grounded retrieval system, the architecture operates under four strict constraints:
1. **Zero External Knowledge:** The model must answer using *only* retrieved text chunks.
2. **Deterministic Citation Provenance:** Gemini output is treated as untrusted data; application code—not the LLM—resolves and formats citation metadata.
3. **Exact Fallback Contract:** If context is insufficient, the system must emit an exact fallback message rather than speculating.
4. **Offline Reproducibility:** Storage, retrieval, and citation validation must run 100% locally without relying on external databases.

## System Architecture & Pipeline Design

The end-to-end architecture is divided into an offline ingestion phase and a real-time query execution phase:

```mermaid
graph TD
    subgraph Offline Ingestion & Vector Indexing
        A[9 Source PDFs / data/raw/] --> B[PyMuPDF Text Extraction]
        B --> C[Page-Bounded Paragraph Chunking / 272 Chunks]
        C --> D[Gemini Embedding 2 / 768-dim]
        D --> E[L2 Normalization & FAISS IndexFlatIP]
        E --> F[Persisted Runtime Store / index/]
    end

    subgraph Query Execution & Citation Trust Boundary
        G[User Question] --> H[Gemini Query Embedding / gemini-embedding-2]
        H --> I[L2 Normalization]
        I --> J[FAISS Top-4 Dense Search]
        J --> K[Ranked Context Chunks + C1..C4 IDs]
        K --> L[Gemini 3.8 Flash Generation]
        L --> M[Raw Answer with C-IDs]
        M --> N[Stage 10 Deterministic Citation Parser]
        N --> O[Streamlit UI Rendered Answer & Source Cards]
    end
```

### Technical Pipeline Specifications
1. **Document Ingestion (Stage 4):** Reads 9 raw PDFs using PyMuPDF (`fitz`), preserving physical page boundaries, validating dataset manifest entries, and extracting 500,649 raw characters.
2. **Text Processing & Chunking (Stage 5):** Performs light normalization (stripping control characters while preserving sentence casing and punctuation) and paragraph-aware chunking (`450` target tokens, `600` max tokens) bounded by physical PDF pages. Produces 272 canonical chunks with deterministic IDs (`{doc_id}_p{page:03d}_c{index:03d}`).
3. **Embedding Generation (Stage 6):** Embeds chunk texts sequentially using Google's `gemini-embedding-2` model at 768 dimensions via the official `google-genai` SDK, verifying finite values and non-zero norms.
4. **Vector Storage & FAISS Indexing (Stage 7):** Constructs an exact `faiss.IndexFlatIP(768)` store over a row-wise L2-normalized `(272, 768)` float32 NumPy matrix. Stores associated chunk metadata and fingerprints index state via SHA-256 manifest files.

## Retrieval System & Vector Storage

The retrieval system delivers relevant context chunks to the generation model while protecting against vector misalignment and quota exhaustion:

1. **Row-Wise L2 Normalization:** Document vectors and query vectors are row-wise L2-normalized using `faiss.normalize_L2()`. Because vectors lie on the unit hypersphere, FAISS inner-product (`IndexFlatIP`) matches exact cosine similarity.
2. **Top-K Retrieval Strategy:** Fixed `Top-K = 4` dense retrieval returns the highest-scoring chunks alongside their rank order (1..4).
3. **Query/Index Compatibility Enforcement:** Prior to invoking API endpoints, the system validates model names (`gemini-embedding-2`) and vector dimensions (`768`) against the stored index manifest.
4. **Retrieval Sanity Set:** An 8-question hand-selected retrieval sanity set confirmed expected source presence in Top-4 for all eight test questions. This engineering check validated index integrity across target documents but does not constitute a general retrieval-accuracy benchmark.

## Grounding & Citation Trust Boundary

A central trust boundary of this application is the **strict separation between LLM generation and citation authority**:

- **The LLM is NOT authoritative for metadata:** Gemini outputs natural-language text interspersed with controlled identifiers (`[C1]`, `[C2]`). It is prohibited from generating footnotes, file paths, or bibliography sections.
- **Deterministic Application Validation:** Stage 10 Python code (`src/citations.py`) intercepts the raw model answer:
  - Parses bracketed markers using regular expressions.
  - Verifies that cited C-IDs exist in the Stage 9 `source_map`.
  - Rejects malformed marker syntax (e.g. `[c1]`, `[C0]`, `[C1; C2]`) or duplicate IDs (`[C1, C1]`).
  - Looks up verified filenames, physical page numbers, titles, publishers, and source URLs from trusted retrieval metadata.
  - Formats inline user-facing citations (e.g. `(bald_eagle_lead_exposure.pdf, p. 1)`).
- **Fail-Closed Safeguards:** If an answer contains invalid citations, unmapped source IDs, or supported claims without markers, the answer is withheld and a user-safe error message is returned.
- **Semantic Entailment Limitation:** Citation provenance validation verifies syntactic marker structure, canonical ID existence, and metadata lookup integrity—it does not establish semantic entailment or guarantee factual correctness. Formal answer-quality evaluation remains planned for Stage 14.

## Engineering Tradeoffs

- **Plain Python vs. Heavy Frameworks:** Using raw Python and native SDKs made control flow, error handling, and citation boundaries 100% explicit without black-box framework abstractions.
- **Flat FAISS vs. Approximate Indexing:** `IndexFlatIP` provides exact nearest-neighbor search for 272 vectors without quantization loss.
- **Tracked Runtime Index Artifacts:** Shipping `index/faiss.index`, `index/chunk_metadata.json`, and `index/index_manifest.json` in Git enables instant deployment on Streamlit Community Cloud without re-embedding the corpus at startup.

## Testing & Verification

The project is verified by **188 automated unit tests** covering document ingestion, text processing, chunking, embeddings, FAISS storage, retrieval, generation, citation resolution, and Streamlit AppTest UI interactions:
- 12 Stage 4 Ingestion tests
- 8 Stage 5 Processing tests
- 11 Stage 5 Chunking tests
- 19 Stage 6 Embedding tests
- 22 Stage 7 Vector Store tests
- 24 Stage 8 Retrieval tests
- 20 Stage 9 Generation tests
- 58 Stage 10 Citation tests
- 14 Stage 11 Web App tests

All 188 tests run 100% offline without live API keys or network requests.

## Deployment & Public Access

The web application is publicly deployed on **Streamlit Community Cloud** (Python 3.13):
- **Live Demo URL:** https://animal-knowledge-rag-bpnfy8iqoxqrbcyeasica2.streamlit.app/
- **Secret Management:** Process environment `GEMINI_API_KEY` configured securely via Streamlit Cloud platform secrets (`st.secrets`).

## Key Lessons Learned

1. **SDK Response Contracts Change:** Relying on precise field access (e.g. `response.embeddings[0].values`) avoids subtle runtime type failures.
2. **Mocking Must Exercise Real Code:** Using Streamlit's native `AppTest` with injected backends proved UI behavior, form bounds, and error mapping without mocking internal framework structures.
3. **Citation Formatting $\neq$ Entailment:** Syntactic marker validation enforces provenance hygiene, but validating semantic entailment requires formal evaluation workflows.

## Project Limitations & Future Roadmap

- **Curated Dataset Scope:** Operating over a focused 9-document demo corpus combining government factsheets and peer-reviewed open-access research (85 physical pages / 272 chunks).
- **Text-Based Processing:** PDFs must contain extractable text; image-based OCR is not currently implemented.
- **Single-Turn Experience:** System answers individual user questions without multi-turn conversational memory.
- **Fixed Retrieval Baseline:** Dense Top-4 retrieval without BM25 hybrid search or reranking.
- **Syntactic Citation Verification:** Citation provenance validation verifies syntactic marker structure and metadata lookup; formal semantic entailment evaluation remains planned for Stage 14.
- **Upcoming Stage 14:** Stage 14 will address formal retrieval and answer evaluation and technical interview preparation; its evaluation design remains to be approved.
