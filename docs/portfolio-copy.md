# Portfolio Snippets & Copy — Animal Knowledge RAG Assistant

## 1. One-Line Description (27 words)
An end-to-end Retrieval-Augmented Generation (RAG) system answering animal science questions with Gemini 3.8 Flash generation, 768-dimensional FAISS vector search, and deterministic application-level citation validation.

## 2. Short Description (90 words)
The Animal Knowledge RAG Assistant is a fact-grounded question-answering application built over a curated 9-document corpus combining government factsheets and peer-reviewed open-access research (85 physical pages). It pairs Google Gemini Embedding 2 (768-dim) vectors and a normalized FAISS IndexFlatIP store with Gemini 3.8 Flash generation. To enforce context constraints and reduce hallucination risk, the system applies strict context-grounding instructions and delegates citation metadata resolution to deterministic Python code. Malformed markers or unsupported claims trigger fail-closed validation or an exact fallback sentence. Verified with 188 automated unit tests and deployed publicly on Streamlit Community Cloud.

## 3. Medium Description (204 words)
The Animal Knowledge RAG Assistant addresses factual grounding in domain-specific AI by implementing a context-constrained Retrieval-Augmented Generation (RAG) architecture over a curated 9-document corpus combining government factsheets and peer-reviewed open-access research.

The engineering pipeline ingests raw scientific PDFs via PyMuPDF, performs page-bounded paragraph chunking (272 chunks across 85 pages), and indexes 768-dimensional `gemini-embedding-2` vectors inside a normalized FAISS `IndexFlatIP` store. At query time, the system normalizes the question vector, retrieves the Top-4 context chunks, and formats them with controlled source identifiers (`C1..C4`). Gemini 3.8 Flash generates concise answers constrained by strict context-grounding instructions.

A central trust boundary of this architecture is the **separation of LLM generation from citation authority**: Gemini outputs controlled marker IDs, but deterministic application code validates citation grammar, verifies canonical source IDs, looks up trusted filename and physical page metadata, and renders user-facing inline references (e.g. `(bald_eagle_lead_exposure.pdf, p. 1)`). Citation provenance validation verifies syntactic marker structure, canonical ID existence, and metadata lookup integrity—it does not establish semantic entailment or guarantee factual correctness.

The application is thoroughly verified with 188 automated unit tests (including offline Streamlit AppTest coverage) and is publicly deployed on Streamlit Community Cloud (Python 3.13).

## 4. Resume Bullets
- Architected a grounded RAG pipeline using Gemini Embedding 2 (768-dim), FAISS IndexFlatIP vector storage (272 chunks across 85 pages), and Gemini 3.8 Flash generation for scientific document QA.
- Implemented a fail-closed Stage 10 citation parser in Python that validates controlled source IDs, looks up trusted page metadata, and renders inline filename/page citations without relying on LLM output.
- Developed a Streamlit web application with native AppTest coverage, URL safety validation, and safe error mapping, verified across 188 automated unit tests and deployed on Streamlit Community Cloud.

## 5. Technology Keyword Line
**Technologies:** Python 3.13, Streamlit 1.64.0, Google Gen AI SDK (`google-genai`), FAISS (`faiss-cpu`), PyMuPDF (`fitz`), NumPy, `unittest`, Streamlit `AppTest`, Git, Streamlit Community Cloud.

## 6. Project Links
- **GitHub Repository:** https://github.com/DPmalaviya/animal-knowledge-rag
- **Live Demo Application:** https://animal-knowledge-rag-bpnfy8iqoxqrbcyeasica2.streamlit.app/
