# Evaluation Framework — Animal Knowledge RAG Assistant

This directory contains the Stage 14 evaluation workflow, frozen golden dataset, metric calculations, and manual review handoff for the **Animal Knowledge RAG Assistant**.

---

## 1. Locked System Architecture

The evaluation measures the existing production baseline without modification:
- **Corpus:** 9 PDFs / 85 physical pages / 272 canonical chunks (`data/raw/`).
- **Ingestion & Chunking:** PyMuPDF text extraction; page-bounded paragraph chunking (`450` target tokens, `600` max tokens, `75` token overlap).
- **Embedding Model:** `gemini-embedding-2` at 768 dimensions via the official `google-genai` SDK.
- **Vector Store:** FAISS `IndexFlatIP` storing row-wise L2-normalized `(272, 768)` float32 vectors.
- **Retrieval Strategy:** Fixed Top-K = 4 dense inner-product search.
- **Generation Model:** `gemini-3.8-flash` with low thinking level and strict context-grounding system instructions.
- **Citation Parser:** Stage 10 deterministic Python resolver parsing controlled `[C1]`..`[C4]` identifiers and mapping trusted metadata (`filename`, `page_number`, `title`, `publisher`, `source_url`).

---

## 2. Frozen Dataset (`evaluation/golden_questions.json`)

The golden evaluation dataset comprises **20 questions**:
- **16 Supported Questions:** Grounded in specific document passages across all 9 approved PDFs and 5 animal groups.
- **4 Unsupported Questions:** Target out-of-corpus topics/species to test exact fallback behavior.
- **Distribution:** 12 newly authored Stage 14 questions, 4 development sanity examples.

### Approved Index Artifact Hashes
```text
index/faiss.index
a5622eb106e81a4cc151d6ed33939239291f1da6df48329fc5ffa0a3b966038e

index/chunk_metadata.json
814cab2575bfe63060321faafc20f971dc11c6a7a59c18dbc75876275c617877

index/index_manifest.json
65ba5bc1a8d1a7bb700e0d614dceb98d9329c2ae5fbc904995a6f0cd3be507ef
```

---

## 3. Evaluation Modules

1. `evaluation/run_evaluation.py`: Runs single-pass Top-4 retrieval and Gemini 3.8 Flash generation for all 20 questions, persisting results to `evaluation/results/`.
2. `evaluation/score_evaluation.py`: Computes deterministic metrics and parses human evaluation scores from `evaluation/results/manual_review.csv`.

---

## 4. Execution Commands

### Run Evaluation Snapshot
```bash
python -m evaluation.run_evaluation
```
*Budget:* 20 query-embedding requests + 20 generation requests = 40 normal API calls.

### Score Results & Generate Summary
```bash
python -m evaluation.score_evaluation
```
Produces `evaluation/results/summary.json` and `evaluation/results/manual_review.csv`.

---

## 5. Metric Formulas

- **Document Hit@4 Rate:** `(Supported questions with at least 1 expected filename in Top-4) / 16`
- **Document MRR@4:** `Mean of (1 / rank_of_first_expected_filename) across 16 supported questions`
- **Page Hit@4 Rate:** `(Supported questions with at least 1 accepted (filename, page) pair in Top-4) / 16`
- **Supported Non-Fallback Rate:** `(Supported questions generating a non-fallback answer) / 16`
- **Exact Fallback Rate:** `(Unsupported questions outputting exact fallback sentence) / 4`
- **Citation Acceptance Rate:** `(Supported non-fallback answers with valid citation IDs and metadata) / (Attempted supported non-fallback answers)`

---

## 6. Human Review Handoff Rubric (0 / 1 / 2)

Per the Stage 14 brief, human review is performed by a human reviewer using `evaluation/results/manual_review.csv`:
- **Correctness:** `0` (incorrect/misleading) | `1` (partially correct) | `2` (fully correct)
- **Groundedness:** `0` (materially unsupported) | `1` (minor unsupported detail) | `2` (fully grounded)
- **Citation Support:** `0` (unsupported citations) | `1` (partially supported) | `2` (fully citation-supported)
