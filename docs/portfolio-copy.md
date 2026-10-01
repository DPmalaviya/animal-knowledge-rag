# Portfolio Snippets & Copy — Animal Knowledge RAG Assistant

## 1. One-Line Description (31 words)
A production-grade Retrieval-Augmented Generation (RAG) system answering animal science questions with Gemini 3.8 Flash generation, 768-dimensional FAISS vector search, and deterministic application-level filename and physical-page citation validation.

## 2. Short Description (98 words)
The Animal Knowledge RAG Assistant is a fact-grounded question-answering application built over a 9-document / 85-physical-page scientific PDF corpus. It pairs Google Gemini Embedding 2 (768-dim) vectors and a normalized FAISS IndexFlatIP store with Gemini 3.8 Flash generation. To eliminate LLM hallucination and unverified citations, the system enforces strict prompt grounding rules and delegates citation metadata resolution to deterministic Python code. Malformed citation markers or unsupported claims trigger fail-closed validation or an exact fallback sentence. Verified with 188 automated unit tests and deployed publicly on Streamlit Community Cloud.

## 3. Medium Description (218 words)
The Animal Knowledge RAG Assistant addresses factual hallucination in scientific AI by implementing a grounded Retrieval-Augmented Generation (RAG) architecture over 9 peer-reviewed animal biology PDFs.

The pipeline ingests raw PDFs via PyMuPDF, performs page-bounded paragraph chunking (272 chunks across 85 pages), and indexes 768-dimensional `gemini-embedding-2` vectors inside a normalized FAISS `IndexFlatIP` store. At query time, the system normalizes the question vector, retrieves the Top-4 context chunks, and formats them with controlled source identifiers (`C1..C4`). Gemini 3.8 Flash generates concise answers constrained by strict grounding rules.

A central architectural innovation is the **separation of LLM generation from citation authority**: Gemini outputs controlled marker IDs, but deterministic application code validates citation grammar, verifies canonical source IDs, looks up trusted filename and page metadata, and renders user-facing inline references (e.g. `(bald_eagle_lead_exposure.pdf, p. 1)`). If context is insufficient or citations fail validation, the system executes a fail-closed policy.

The application is thoroughly verified with 188 automated unit tests (including offline Streamlit AppTest coverage) and is deployed on Streamlit Community Cloud (Python 3.13).

## 4. Resume Bullets
- Architected a grounded RAG pipeline using Gemini Embedding 2 (768-dim), FAISS IndexFlatIP vector storage (272 chunks across 85 pages), and Gemini 3.8 Flash generation for scientific document QA.
- Implemented a fail-closed Stage 10 citation parser in Python that validates controlled source IDs, looks up trusted page metadata, and renders inline filename/page citations without relying on LLM output.
- Developed a Streamlit web application with native AppTest coverage, URL safety validation, and safe error mapping, verified across 188 automated unit tests and deployed on Streamlit Community Cloud.

## 5. Technology Keyword Line
**Technologies:** Python 3.13, Streamlit 1.64.0, Google Gen AI SDK (`google-genai`), FAISS (`faiss-cpu`), PyMuPDF (`fitz`), NumPy, `unittest`, Streamlit `AppTest`, Git, Streamlit Community Cloud.

## 6. Project Links
- **GitHub Repository:** https://github.com/DPmalaviya/animal-knowledge-rag
- **Live Demo Application:** https://animal-knowledge-rag-bpnfy8iqoxqrbcyeasica2.streamlit.app/
