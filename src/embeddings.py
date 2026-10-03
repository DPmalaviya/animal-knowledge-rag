"""Embeddings module for Animal Knowledge RAG Assistant.

Generates 768-dimensional document chunk embeddings using the official Google Gen AI SDK
and the gemini-embedding-2 model.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    errors = None  # type: ignore


DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIMENSION = 768
DEFAULT_PROVIDER_TIMEOUT_MS = 20_000
SDK_HTTP_RETRY_ATTEMPTS = 1


def prepare_document_text(chunk: Dict[str, Any]) -> str:
    """Format document chunk into the approved semantic embedding input string.

    Format: title: {title} | text: {chunk_text}

    Args:
        chunk: Stage 5 chunk record dictionary.

    Returns:
        Formatted semantic input string.

    Raises:
        ValueError: If chunk text or title is missing.
    """
    title = (chunk.get("title") or "").strip()
    text = (chunk.get("text") or "").strip()

    if not text:
        raise ValueError(f"Chunk '{chunk.get('chunk_id')}' has missing or empty text.")
    if not title:
        raise ValueError(f"Chunk '{chunk.get('chunk_id')}' has missing or empty title.")

    return f"title: {title} | text: {text}"


def create_genai_client(api_key: Optional[str] = None) -> Any:
    """Create and return an official Google Gen AI SDK client.

    Args:
        api_key: Optional explicit API key. If not provided, reads GEMINI_API_KEY from environment.

    Returns:
        Configured genai.Client instance.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or empty.
        ImportError: If google-genai package is not installed.
    """
    if genai is None:
        raise ImportError(
            "The 'google-genai' package is required for embeddings. "
            "Please install it using 'pip install google-genai'."
        )

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable "
            "in your process environment before running live embeddings."
        )

    http_options = types.HttpOptions(
        timeout=DEFAULT_PROVIDER_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=SDK_HTTP_RETRY_ATTEMPTS),
    )
    return genai.Client(api_key=key.strip(), http_options=http_options)


def calculate_l2_norm(vector: List[float]) -> float:
    """Calculate the L2 norm (Euclidean length) of a vector.

    Args:
        vector: List of floating point values.

    Returns:
        L2 norm as float.
    """
    return math.sqrt(sum(x * x for x in vector))


def validate_embedding_vector(vector: Any, expected_dim: int = DEFAULT_DIMENSION) -> List[float]:
    """Validate that an embedding vector is finite, non-boolean, numeric, and correct-dimensioned.

    Args:
        vector: Returned embedding values.
        expected_dim: Expected vector dimension (default 768).

    Returns:
        Validated list of float values.

    Raises:
        ValueError: If validation fails.
    """
    if not isinstance(vector, (list, tuple)):
        raise ValueError(f"Invalid embedding type: expected list, got {type(vector).__name__}")

    if len(vector) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )

    validated = []
    for idx, val in enumerate(vector):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"Embedding vector element at index {idx} is non-numeric: {val} ({type(val).__name__})"
            )
        float_val = float(val)
        if not math.isfinite(float_val):
            raise ValueError(f"Embedding vector element at index {idx} is non-finite: {float_val}")
        validated.append(float_val)

    norm = calculate_l2_norm(validated)
    if norm <= 1e-6:
        raise ValueError(f"Embedding vector has zero or near-zero L2 norm: {norm}")

    return validated


def embed_single_chunk(
    client: Any,
    chunk: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """Embed a single document chunk using the Gemini API embed_content model.

    Args:
        client: Configured GenAI client.
        chunk: Stage 5 chunk record dictionary.
        model_name: Gemini embedding model name (default "gemini-embedding-2").
        dimension: Desired output dimension (default 768).
        max_retries: Maximum retries for transient failures.
        retry_delay: Initial retry delay in seconds.

    Returns:
        New chunk embedding record dictionary containing all original fields plus embedding metadata.

    Raises:
        ValueError: If formatting, request, or validation fails.
        Exception: If API call fails non-transiently or retries are exhausted.
    """
    prep_text = prepare_document_text(chunk)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            config = types.EmbedContentConfig(output_dimensionality=dimension)
            response = client.models.embed_content(
                model=model_name,
                contents=prep_text,
                config=config,
            )

            if not hasattr(response, "embeddings") or not response.embeddings:
                raise ValueError("API response missing 'embeddings' field or embeddings list is empty.")

            if len(response.embeddings) != 1:
                raise ValueError(
                    f"API response returned {len(response.embeddings)} embeddings; expected exactly 1."
                )

            single_emb = response.embeddings[0]
            raw_values = getattr(single_emb, "values", None)
            if raw_values is None:
                raise ValueError("API response embedding object missing 'values' field.")

            validated_vector = validate_embedding_vector(raw_values, expected_dim=dimension)

            # Create new record to avoid mutating original chunk input
            new_record = dict(chunk)
            new_record["embedding"] = validated_vector
            new_record["embedding_model"] = model_name
            new_record["embedding_dimension"] = dimension

            return new_record

        except Exception as e:
            last_error = e
            err_msg = str(e).lower()

            # Fail immediately on non-transient auth/permission/invalid argument errors
            if any(k in err_msg for k in ["invalid_argument", "unauthorized", "forbidden", "api_key", "401", "403"]):
                raise ValueError(f"Non-transient API authentication or permission error: {e}") from e

            if attempt < max_retries:
                sleep_time = retry_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    raise ValueError(
        f"Exhausted retries ({max_retries}) embedding chunk '{chunk.get('chunk_id')}': {last_error}"
    ) from last_error


def embed_corpus_chunks(
    chunks: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL,
    dimension: int = DEFAULT_DIMENSION,
    delay_between_requests: float = 0.1,
) -> List[Dict[str, Any]]:
    """Embed a list of Stage 5 chunk records sequentially into 768-dimensional vectors.

    Args:
        chunks: List of Stage 5 chunk record dictionaries.
        client: Optional pre-configured GenAI client. If None, creates client.
        model_name: Gemini model identifier.
        dimension: Output vector dimension.
        delay_between_requests: Throttle delay between sequential requests in seconds.

    Returns:
        List of new chunk embedding records.

    Raises:
        ValueError: If input validation fails or any chunk embedding fails.
    """
    if not chunks:
        raise ValueError("Input chunks collection is empty.")

    # Validate input chunk IDs uniqueness
    seen_ids = set()
    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            raise ValueError("Found chunk record missing 'chunk_id'.")
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id in input collection: '{cid}'")
        seen_ids.add(cid)

    active_client = client or create_genai_client()
    embedding_records = []

    for idx, chunk in enumerate(chunks, start=1):
        record = embed_single_chunk(
            active_client,
            chunk,
            model_name=model_name,
            dimension=dimension,
        )
        embedding_records.append(record)

        if delay_between_requests > 0 and idx < len(chunks):
            time.sleep(delay_between_requests)

    # Validate output integrity against input collection
    if len(embedding_records) != len(chunks):
        raise ValueError(
            f"Embedding collection size mismatch: expected {len(chunks)}, got {len(embedding_records)}"
        )

    for in_c, out_c in zip(chunks, embedding_records):
        if in_c["chunk_id"] != out_c["chunk_id"]:
            raise ValueError(
                f"Embedding output ID mismatch: expected '{in_c['chunk_id']}', got '{out_c['chunk_id']}'"
            )

    return embedding_records


def save_embeddings_artifact(
    records: List[Dict[str, Any]],
    artifact_path: str = "data/processed/chunk_embeddings.json",
) -> None:
    """Save completed embedding records atomically to a JSON artifact file.

    Args:
        records: List of chunk embedding records.
        artifact_path: Path to target JSON file.
    """
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f".{path.name}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    tmp_path.replace(path)


def load_embeddings_artifact(
    artifact_path: str = "data/processed/chunk_embeddings.json",
) -> List[Dict[str, Any]]:
    """Load and validate an existing chunk embeddings JSON artifact.

    Args:
        artifact_path: Path to JSON artifact file.

    Returns:
        List of loaded and validated chunk embedding records.

    Raises:
        FileNotFoundError: If artifact file does not exist.
        ValueError: If JSON file validation fails.
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings artifact file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError("Loaded embeddings artifact is not a JSON list.")

    seen_ids = set()
    for idx, r in enumerate(records):
        if "chunk_id" not in r or "embedding" not in r:
            raise ValueError(f"Record at index {idx} missing required fields.")
        cid = r["chunk_id"]
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id in artifact: '{cid}'")
        seen_ids.add(cid)

        validate_embedding_vector(r["embedding"], expected_dim=r.get("embedding_dimension", DEFAULT_DIMENSION))

    return records


def main() -> None:
    """CLI entrypoint for embedding execution and verification."""
    from src.chunking import chunk_pages
    from src.ingestion import ingest_corpus
    from src.processing import process_pages

    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lstrip("-")

    print(f"Starting Stage 6 Embeddings Pipeline (mode: {mode})...")

    # 1. Regenerate Stage 5 chunks
    raw_pages = ingest_corpus()
    proc_pages = process_pages(raw_pages)
    chunks = chunk_pages(proc_pages)

    print(f"Loaded {len(chunks)} chunks across {len(set(c['document_id'] for c in chunks))} documents.")

    client = create_genai_client()

    if mode == "smoke":
        target_chunks = chunks[:1]
    elif mode == "sample":
        # Select 3 representative chunks from factsheet, PLOS, Frontiers
        sample_files = [
            "bald_eagle_factsheet.pdf",
            "bald_eagle_lead_exposure.pdf",
            "sea_turtle_foraging.pdf",
        ]
        target_chunks = []
        for fn in sample_files:
            match = next((c for c in chunks if c["filename"] == fn), None)
            if match:
                target_chunks.append(match)
        if not target_chunks:
            target_chunks = chunks[:3]
    else:
        target_chunks = chunks

    print(f"Embedding {len(target_chunks)} chunk(s) via Gemini API ({DEFAULT_MODEL}, {DEFAULT_DIMENSION}-dim)...")
    embedded_records = embed_corpus_chunks(target_chunks, client=client)

    # Calculate L2 norms
    norms = [calculate_l2_norm(r["embedding"]) for r in embedded_records]
    min_norm, max_norm, mean_norm = min(norms), max(norms), sum(norms) / len(norms)

    print("\n" + "=" * 80)
    print("STAGE 6 EMBEDDINGS SUMMARY")
    print("=" * 80)
    print(f"Model Requested:          {DEFAULT_MODEL}")
    print(f"Output Dimension:         {DEFAULT_DIMENSION}")
    print(f"Chunks Embedded:          {len(embedded_records)}")
    print(f"Unique Chunk IDs:         {len(set(r['chunk_id'] for r in embedded_records))}")
    print(f"Vector Value Integrity:   100% finite numeric floats")
    print(f"L2 Norm Statistics:       Min={min_norm:.4f}, Max={max_norm:.4f}, Mean={mean_norm:.4f}")
    print("=" * 80)

    if mode == "full" and len(embedded_records) == len(chunks):
        save_embeddings_artifact(embedded_records)
        print("\nSaved full corpus embeddings artifact to: data/processed/chunk_embeddings.json")

    print("\nStage 6 embeddings completed successfully!")


if __name__ == "__main__":
    main()
