"""Retrieval System module for Animal Knowledge RAG Assistant.

Implements semantic Top-K document chunk retrieval using Gemini API embeddings
and local FAISS IndexFlatIP vector search.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    errors = None  # type: ignore

from src.embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    create_genai_client,
    validate_embedding_vector,
)
from src.vector_store import DEFAULT_INDEX_DIR, load_vector_store

DEFAULT_TOP_K = 4
QUERY_PREFIX = "task: question answering | query: "


def prepare_query_text(question: str) -> str:
    """Format raw question string into the approved semantic query embedding input string.

    Format: task: question answering | query: {question}

    Args:
        question: Input user question.

    Returns:
        Formatted semantic query string.

    Raises:
        ValueError: If question is missing, non-string, or whitespace-only.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    # Preserve exact question string inside prefix without modifying characters
    return f"{QUERY_PREFIX}{question}"


def validate_top_k(top_k: Any, max_available: int) -> int:
    """Validate requested Top-K parameter.

    Args:
        top_k: Top-K value to validate.
        max_available: Total number of vectors available in the index.

    Returns:
        Validated integer top_k.

    Raises:
        ValueError: If top_k is invalid or out of range.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be a positive integer, got {type(top_k).__name__}: {top_k}")

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")

    if top_k > max_available:
        raise ValueError(
            f"top_k ({top_k}) exceeds total available index vectors ({max_available})."
        )

    return top_k


def embed_query(
    client: Any,
    prepared_query_text: str,
    model_name: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
) -> List[float]:
    """Embed a prepared query string using the Gemini API embed_content model.

    Args:
        client: Configured GenAI client instance.
        prepared_query_text: Formatted query input string.
        model_name: Gemini model identifier.
        dimension: Desired output dimension (default 768).

    Returns:
        Validated 768-dimensional float embedding vector.

    Raises:
        ValueError: If response contract or vector validation fails.
    """
    if types is None:
        raise ImportError("google-genai package is required for query embedding.")

    config = types.EmbedContentConfig(output_dimensionality=dimension)
    try:
        response = client.models.embed_content(
            model=model_name,
            contents=prepared_query_text,
            config=config,
        )
    except Exception as e:
        err_msg = str(e).lower()
        if any(k in err_msg for k in ["invalid_argument", "unauthorized", "forbidden", "api_key", "401", "403"]):
            raise ValueError(f"Non-transient API authentication or permission error embedding query: {e}") from e
        raise ValueError(f"API request failed embedding query: {e}") from e

    if not hasattr(response, "embeddings") or not response.embeddings:
        raise ValueError("API response missing 'embeddings' field or embeddings list is empty.")

    if len(response.embeddings) != 1:
        raise ValueError(
            f"API response returned {len(response.embeddings)} embeddings; expected exactly 1."
        )

    single_emb = response.embeddings[0]
    raw_values = getattr(single_emb, "values", None)
    if raw_values is None:
        raise ValueError("API response query embedding object missing 'values' field.")

    return validate_embedding_vector(raw_values, expected_dim=dimension)


def prepare_query_matrix(
    vector: List[float],
    expected_dim: int = DEFAULT_DIMENSION,
) -> np.ndarray:
    """Convert and defensively L2-normalize a query embedding vector into a 2D float32 matrix.

    Args:
        vector: 768-element list of float values.
        expected_dim: Expected dimension (default 768).

    Returns:
        C-contiguous float32 2D NumPy array of shape (1, expected_dim).

    Raises:
        ValueError: If vector, matrix, or normalization validation fails.
    """
    validated_vec = validate_embedding_vector(vector, expected_dim=expected_dim)
    try:
        matrix = np.array([validated_vec], dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to convert query vector to float32 NumPy matrix: {e}") from e

    if matrix.shape != (1, expected_dim):
        raise ValueError(f"Query matrix shape invalid: expected (1, {expected_dim}), got {matrix.shape}")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Converted query float32 matrix contains non-finite values.")

    pre_norm = float(np.linalg.norm(matrix))
    if pre_norm <= 1e-6:
        raise ValueError(f"Query matrix has zero or near-zero pre-normalization L2 norm: {pre_norm}")

    # Ensure C-contiguity
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    # Defensive L2-normalization
    faiss.normalize_L2(matrix)

    post_norm = float(np.linalg.norm(matrix))
    if not (0.999 <= post_norm <= 1.001):
        raise ValueError(f"Post-normalization query matrix norm out of tolerance [0.999, 1.001]: {post_norm}")

    return matrix


def search_by_vector(
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    query_matrix: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Perform FAISS IndexFlatIP inner product search and map results to chunk metadata.

    Args:
        index: Loaded FAISS IndexFlatIP instance.
        metadata: List of loaded chunk metadata dictionaries.
        query_matrix: C-contiguous normalized float32 query matrix of shape (1, 768).
        top_k: Number of top results to return (default 4).

    Returns:
        List of Top-K result dictionaries ordered by similarity rank.

    Raises:
        ValueError: If search parameters, shapes, or mapping checks fail.
    """
    validate_top_k(top_k, max_available=index.ntotal)

    if not isinstance(query_matrix, np.ndarray) or query_matrix.shape != (1, index.d):
        raise ValueError(
            f"Query matrix shape invalid for FAISS search: expected (1, {index.d}), got {query_matrix.shape}"
        )

    if query_matrix.dtype != np.float32:
        query_matrix = np.ascontiguousarray(query_matrix, dtype=np.float32)

    # Execute FAISS search
    scores, ids = index.search(query_matrix, top_k)

    if scores.shape != (1, top_k) or ids.shape != (1, top_k):
        raise ValueError(f"FAISS search output shape mismatch: expected (1, {top_k})")

    results = []
    seen_ids = set()

    for rank_idx in range(top_k):
        raw_id = ids[0][rank_idx]
        raw_score = scores[0][rank_idx]

        if not np.isfinite(raw_score):
            raise ValueError(f"FAISS search returned non-finite score at rank {rank_idx+1}: {raw_score}")

        faiss_id = int(raw_id)
        if faiss_id < 0 or faiss_id >= index.ntotal:
            raise ValueError(
                f"FAISS search returned invalid or out-of-range ID at rank {rank_idx+1}: {faiss_id}"
            )

        if faiss_id in seen_ids:
            raise ValueError(f"FAISS search returned duplicate ID in Top-K results: {faiss_id}")
        seen_ids.add(faiss_id)

        meta = metadata[faiss_id]
        if meta.get("faiss_id") != faiss_id:
            raise ValueError(
                f"Metadata mapping position mismatch: meta faiss_id={meta.get('faiss_id')}, expected {faiss_id}"
            )

        # Build clean result entry preserving all source metadata
        result_entry = {
            "rank": rank_idx + 1,
            "faiss_id": faiss_id,
            "similarity_score": float(raw_score),
            "chunk_id": meta["chunk_id"],
            "document_id": meta["document_id"],
            "filename": meta["filename"],
            "page_number": meta["page_number"],
            "chunk_index": meta["chunk_index"],
            "text": meta["text"],  # Full un-truncated chunk text preserved for Stage 9 RAG
            "title": meta["title"],
            "publisher": meta["publisher"],
            "source_url": meta["source_url"],
            "estimated_token_count": meta["estimated_token_count"],
            "embedding_model": meta["embedding_model"],
            "embedding_dimension": meta["embedding_dimension"],
        }
        results.append(result_entry)

    # Verify scores are non-increasing
    for r in range(len(results) - 1):
        if results[r]["similarity_score"] < results[r + 1]["similarity_score"] - 1e-6:
            raise ValueError(
                f"Top-K similarity scores are not non-increasing between rank {r+1} and {r+2}"
            )

    return results


def retrieve(
    question: str,
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = DEFAULT_TOP_K,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
) -> List[Dict[str, Any]]:
    """High-level semantic Top-K document retrieval pipeline.

    Validates prerequisites (question formatting, index loading, top_k range)
    prior to invoking the Gemini API embedding endpoint.

    Args:
        question: Raw user question string.
        index_dir: Directory containing Stage 7 index artifacts (default "index").
        top_k: Number of results to retrieve (default 4).
        api_key: Optional explicit API key.
        client: Optional pre-configured GenAI client instance.
        model_name: Gemini embedding model name (default "gemini-embedding-2").
        dimension: Output vector dimension (default 768).

    Returns:
        List of Top-K retrieved result dictionaries.

    Raises:
        ValueError: If prerequisite checks, embedding, or search fails.
    """
    # 1. Validate question prerequisite
    prepared_query = prepare_query_text(question)

    # 2. Validate & load Stage 7 vector store prerequisite
    index, metadata, manifest = load_vector_store(index_dir)

    # 3. Validate top_k prerequisite against index ntotal
    validate_top_k(top_k, max_available=index.ntotal)

    # 4. Create client & request query embedding
    active_client = client or create_genai_client(api_key=api_key)
    query_vector = embed_query(
        active_client,
        prepared_query,
        model_name=model_name,
        dimension=dimension,
    )

    # 5. Prepare normalized query matrix
    query_matrix = prepare_query_matrix(query_vector, expected_dim=dimension)

    # 6. Execute search and map metadata
    return search_by_vector(index, metadata, query_matrix, top_k=top_k)


def format_cli_results(
    question: str,
    results: List[Dict[str, Any]],
    preview_length: int = 120,
) -> str:
    """Format Top-K retrieval results into readable CLI text preview output.

    Args:
        question: Original raw user question string.
        results: List of Top-K result dictionaries.
        preview_length: Character limit for text preview snippet.

    Returns:
        Formatted multi-line CLI text output.
    """
    lines = [
        "=" * 80,
        "STAGE 8 SEMANTIC RETRIEVAL RESULTS",
        "=" * 80,
        f"Query: \"{question}\"",
        f"Top-K Results: {len(results)}",
        "-" * 80,
    ]

    for r in results:
        text_snippet = r['text'].replace('\n', ' ')
        if len(text_snippet) > preview_length:
            text_snippet = text_snippet[:preview_length] + "..."

        lines.extend([
            f"Rank #{r['rank']} | Similarity Score: {r['similarity_score']:.6f}",
            f"  Chunk ID:   {r['chunk_id']}",
            f"  Filename:   {r['filename']} (Page {r['page_number']})",
            f"  Title:      {r['title']}",
            f"  Snippet:    \"{text_snippet}\"",
            "-" * 80,
        ])

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for Stage 8 Retrieval System execution."""
    parser = argparse.ArgumentParser(
        description="Animal Knowledge RAG Assistant — Stage 8 Semantic Retrieval CLI"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Question string to retrieve context for.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top document chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Path to FAISS index directory (default: {DEFAULT_INDEX_DIR}).",
    )

    args = parser.parse_args()

    print(f"Executing Stage 8 Retrieval for query: \"{args.query}\" (top_k={args.top_k})...")
    try:
        results = retrieve(
            question=args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
        )
        print("\n" + format_cli_results(args.query, results))
        print("Stage 8 Retrieval completed successfully!")
    except Exception as e:
        print(f"\nERROR: Retrieval failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
