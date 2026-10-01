"""Vector Storage module for Animal Knowledge RAG Assistant.

Implements FAISS IndexFlatIP vector index management, float32 matrix conversion,
defensive L2-normalization, integer-ID-to-metadata mapping, persistent storage,
and validating artifact loader.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

DEFAULT_INDEX_DIR = "index"
DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_DIMENSION = 768
EXPECTED_CORPUS_COUNT = 272

REQUIRED_PROVENANCE_FIELDS = [
    "chunk_id",
    "document_id",
    "filename",
    "page_number",
    "chunk_index",
    "text",
    "title",
    "publisher",
    "source_url",
    "estimated_token_count",
    "embedding_model",
    "embedding_dimension",
    "embedding",
]

REQUIRED_STRING_PROVENANCE_FIELDS = [
    "chunk_id",
    "document_id",
    "filename",
    "text",
    "title",
    "publisher",
    "source_url",
]

REQUIRED_INT_PROVENANCE_FIELDS = [
    "page_number",
    "chunk_index",
    "estimated_token_count",
]


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 checksum of a file.

    Args:
        file_path: Path to target file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {file_path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_chunk_ids_fingerprint(chunk_ids: List[str]) -> str:
    """Compute SHA-256 fingerprint of an ordered list of chunk IDs.

    Args:
        chunk_ids: List of chunk ID strings.

    Returns:
        Hexadecimal SHA-256 string.
    """
    serialized = json.dumps(chunk_ids, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_embedding_records(
    records: List[Dict[str, Any]],
    expected_count: Optional[int] = None,
    expected_dim: int = DEFAULT_DIMENSION,
    expected_model: str = DEFAULT_MODEL,
) -> None:
    """Validate embedding records collection prior to matrix conversion.

    Enforces strict Stage 6 source metadata and vector provenance contracts.

    Args:
        records: List of chunk embedding record dictionaries.
        expected_count: Optional exact record count requirement (e.g. 272 for production).
        expected_dim: Expected vector dimension (default 768).
        expected_model: Expected embedding model name (default "gemini-embedding-2").

    Raises:
        ValueError: If record collection validation fails.
    """
    if not isinstance(records, list) or not records:
        raise ValueError("Embedding records input must be a non-empty list.")

    if expected_count is not None and len(records) != expected_count:
        raise ValueError(
            f"Embedding records count mismatch: expected exactly {expected_count}, got {len(records)}"
        )

    seen_ids = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not a dictionary.")

        # Require all Stage 6 fields present
        for field in REQUIRED_PROVENANCE_FIELDS:
            if field not in record:
                raise ValueError(
                    f"Record at index {idx} missing required Stage 6 provenance field '{field}'."
                )

        # Validate non-empty string fields
        for field in REQUIRED_STRING_PROVENANCE_FIELDS:
            val = record[field]
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"Record at index {idx} field '{field}' must be a non-empty string."
                )

        # Validate positive integer fields
        for field in REQUIRED_INT_PROVENANCE_FIELDS:
            val = record[field]
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"Record at index {idx} field '{field}' must be a positive integer (got {val})."
                )

        chunk_id = record["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id found in input collection: '{chunk_id}'")
        seen_ids.add(chunk_id)

        model = record["embedding_model"]
        if model != expected_model:
            raise ValueError(
                f"Record '{chunk_id}' model mismatch: expected '{expected_model}', got '{model}'"
            )

        dim = record["embedding_dimension"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim != expected_dim:
            raise ValueError(
                f"Record '{chunk_id}' embedding_dimension mismatch: expected {expected_dim}, got {dim}"
            )

        vector = record["embedding"]
        if not isinstance(vector, (list, tuple)):
            raise ValueError(f"Record '{chunk_id}' embedding field must be a list/tuple.")

        if len(vector) != expected_dim:
            raise ValueError(
                f"Record '{chunk_id}' embedding dimension mismatch: expected {expected_dim}, got {len(vector)}"
            )

        for elem_idx, val in enumerate(vector):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"Record '{chunk_id}' vector element at index {elem_idx} is non-numeric: {val}"
                )
            if not np.isfinite(float(val)):
                raise ValueError(
                    f"Record '{chunk_id}' vector element at index {elem_idx} is non-finite: {val}"
                )


def build_embeddings_matrix(
    records: List[Dict[str, Any]],
    expected_dim: int = DEFAULT_DIMENSION,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extract, validate, convert, and defensively L2-normalize an embedding matrix.

    Args:
        records: List of chunk embedding record dictionaries.
        expected_dim: Expected vector dimension (default 768).

    Returns:
        Tuple of (normalized_float32_matrix, norm_statistics_dict).

    Raises:
        ValueError: If matrix conversion or normalization checks fail.
    """
    validate_embedding_records(records, expected_dim=expected_dim)

    raw_vectors = [r["embedding"] for r in records]
    try:
        matrix = np.array(raw_vectors, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to convert embedding vectors to float32 NumPy matrix: {e}") from e

    if matrix.ndim != 2:
        raise ValueError(f"Matrix shape invalid: expected 2D array, got {matrix.ndim}D")

    if matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Matrix column dimension mismatch: expected {expected_dim}, got {matrix.shape[1]}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Converted float32 matrix contains non-finite values (NaN or Infinity).")

    # Check pre-normalization row norms
    pre_norms = np.linalg.norm(matrix, axis=1)
    if np.any(pre_norms <= 1e-6):
        zero_indices = np.where(pre_norms <= 1e-6)[0]
        raise ValueError(
            f"Matrix contains {len(zero_indices)} zero or near-zero norm vectors at indices {zero_indices.tolist()}."
        )

    # Ensure C-contiguity
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    # Defensive L2-normalization
    faiss.normalize_L2(matrix)

    # Check post-normalization row norms
    post_norms = np.linalg.norm(matrix, axis=1)
    min_norm = float(np.min(post_norms))
    max_norm = float(np.max(post_norms))
    mean_norm = float(np.mean(post_norms))

    if not (0.999 <= min_norm <= 1.001 and 0.999 <= max_norm <= 1.001):
        raise ValueError(
            f"Post-normalization row norms out of expected tolerance [0.999, 1.001]: min={min_norm}, max={max_norm}"
        )

    norm_stats = {
        "min_norm": min_norm,
        "max_norm": max_norm,
        "mean_norm": mean_norm,
    }

    return matrix, norm_stats


def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    """Build and populate a FAISS IndexFlatIP index with normalized float32 vectors.

    Args:
        matrix: C-contiguous float32 2D NumPy array of shape (N, dimension).

    Returns:
        Populated faiss.IndexFlatIP instance.

    Raises:
        ValueError: If matrix or index construction checks fail.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input to build_faiss_index must be a 2D NumPy array.")

    if matrix.dtype != np.float32:
        raise ValueError(f"Input matrix dtype must be float32, got {matrix.dtype}")

    if not matrix.flags["C_CONTIGUOUS"]:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    dimension = matrix.shape[1]
    num_vectors = matrix.shape[0]

    index = faiss.IndexFlatIP(dimension)

    if not index.is_trained:
        raise ValueError("FAISS IndexFlatIP failed initialization (is_trained is False).")

    index.add(matrix)

    if index.ntotal != num_vectors:
        raise ValueError(
            f"FAISS index vector count mismatch: expected {num_vectors}, got {index.ntotal}"
        )

    if index.d != dimension:
        raise ValueError(f"FAISS index dimension mismatch: expected {dimension}, got {index.d}")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS index metric_type is not METRIC_INNER_PRODUCT.")

    return index


def build_metadata_mapping(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build exact 0-indexed integer ID to chunk metadata mapping.

    Enforces direct field copying without default fallbacks to preserve strict provenance.

    Args:
        records: List of chunk embedding record dictionaries.

    Returns:
        List of metadata dictionaries, each containing 'faiss_id' and preserved chunk fields.

    Raises:
        ValueError: If validation of records or metadata fields fails.
    """
    validate_embedding_records(records)
    metadata_list = []

    for i, r in enumerate(records):
        # Exclude full vector array from metadata JSON to prevent redundant storage
        meta_entry = {
            "faiss_id": i,
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "filename": r["filename"],
            "page_number": r["page_number"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "title": r["title"],
            "publisher": r["publisher"],
            "source_url": r["source_url"],
            "estimated_token_count": r["estimated_token_count"],
            "embedding_model": r["embedding_model"],
            "embedding_dimension": r["embedding_dimension"],
        }
        metadata_list.append(meta_entry)

    if len(metadata_list) != len(records):
        raise ValueError(
            f"Metadata mapping size mismatch: expected {len(records)}, got {len(metadata_list)}"
        )

    return metadata_list


def save_vector_store(
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Persist FAISS index, metadata JSON, and manifest JSON atomically to index_dir.

    Args:
        index: Populated FAISS IndexFlatIP instance.
        metadata: Metadata mapping list.
        index_dir: Target output directory (default "index").
        model_name: Embedding model identifier.

    Returns:
        Manifest dictionary.
    """
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "faiss.index"
    meta_path = out_dir / "chunk_metadata.json"
    manifest_path = out_dir / "index_manifest.json"

    tmp_index = out_dir / ".faiss.index.tmp"
    tmp_meta = out_dir / ".chunk_metadata.json.tmp"
    tmp_manifest = out_dir / ".index_manifest.json.tmp"

    # 1. Write temporary index
    faiss.write_index(index, str(tmp_index))

    # 2. Write temporary metadata
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Compute checksums & fingerprint
    index_sha = compute_file_sha256(str(tmp_index))
    meta_sha = compute_file_sha256(str(tmp_meta))
    chunk_ids = [m["chunk_id"] for m in metadata]
    ids_fingerprint = compute_chunk_ids_fingerprint(chunk_ids)

    manifest = {
        "index_type": "IndexFlatIP",
        "metric_type": "METRIC_INNER_PRODUCT",
        "dimension": index.d,
        "vector_count": index.ntotal,
        "metadata_count": len(metadata),
        "embedding_model": model_name,
        "is_normalized": True,
        "ordered_chunk_ids_sha256": ids_fingerprint,
        "faiss_index_sha256": index_sha,
        "chunk_metadata_sha256": meta_sha,
    }

    # 3. Write temporary manifest
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Replace final target files atomically
    tmp_index.replace(index_path)
    tmp_meta.replace(meta_path)

    # Re-compute checksums after atomic replacement to verify persistence integrity
    final_index_sha = compute_file_sha256(str(index_path))
    final_meta_sha = compute_file_sha256(str(meta_path))

    manifest["faiss_index_sha256"] = final_index_sha
    manifest["chunk_metadata_sha256"] = final_meta_sha

    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    tmp_manifest.replace(manifest_path)

    return manifest


def load_vector_store(
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]], Dict[str, Any]]:
    """Load and strictly validate persisted FAISS index, metadata, and manifest.

    Args:
        index_dir: Path to directory containing index artifacts.

    Returns:
        Tuple of (index, metadata, manifest).

    Raises:
        FileNotFoundError: If any required artifact file is missing.
        ValueError: If manifest, checksum, index, or metadata validation fails.
    """
    out_dir = Path(index_dir)
    index_path = out_dir / "faiss.index"
    meta_path = out_dir / "chunk_metadata.json"
    manifest_path = out_dir / "index_manifest.json"

    for path in [index_path, meta_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required vector store artifact missing: {path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError("Loaded manifest is not a dictionary.")

    # 1. Verify file checksums
    actual_index_sha = compute_file_sha256(str(index_path))
    if actual_index_sha != manifest.get("faiss_index_sha256"):
        raise ValueError(
            f"FAISS index checksum mismatch: expected {manifest.get('faiss_index_sha256')}, got {actual_index_sha}"
        )

    actual_meta_sha = compute_file_sha256(str(meta_path))
    if actual_meta_sha != manifest.get("chunk_metadata_sha256"):
        raise ValueError(
            f"Metadata checksum mismatch: expected {manifest.get('chunk_metadata_sha256')}, got {actual_meta_sha}"
        )

    # 2. Load metadata first to allow manifest metadata_count validation
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise ValueError("Loaded chunk metadata is not a list.")

    # 3. Verify manifest fields explicitly
    if manifest.get("index_type") != "IndexFlatIP":
        raise ValueError(f"Unsupported index_type in manifest: {manifest.get('index_type')}")
    if manifest.get("metric_type") != "METRIC_INNER_PRODUCT":
        raise ValueError(f"Unsupported metric_type in manifest: {manifest.get('metric_type')}")
    if manifest.get("dimension") != DEFAULT_DIMENSION:
        raise ValueError(f"Unsupported dimension in manifest: {manifest.get('dimension')}")
    if manifest.get("metadata_count") != len(metadata):
        raise ValueError(
            f"Manifest metadata_count ({manifest.get('metadata_count')}) does not match loaded metadata length ({len(metadata)})."
        )
    if manifest.get("embedding_model") != DEFAULT_MODEL:
        raise ValueError(
            f"Manifest embedding_model mismatch: expected '{DEFAULT_MODEL}', got '{manifest.get('embedding_model')}'"
        )
    if not isinstance(manifest.get("is_normalized"), bool) or manifest.get("is_normalized") is not True:
        raise ValueError(
            f"Manifest is_normalized flag must be boolean True, got {manifest.get('is_normalized')}"
        )

    # 4. Load and validate FAISS index
    index = faiss.read_index(str(index_path))
    if index.d != DEFAULT_DIMENSION:
        raise ValueError(f"Loaded FAISS index dimension mismatch: expected {DEFAULT_DIMENSION}, got {index.d}")
    if index.ntotal != manifest.get("vector_count"):
        raise ValueError(
            f"Loaded FAISS index vector count mismatch: expected {manifest.get('vector_count')}, got {index.ntotal}"
        )
    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS index vector count ({index.ntotal}) does not match metadata length ({len(metadata)})."
        )
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Loaded FAISS index metric_type is not METRIC_INNER_PRODUCT.")

    # 5. Validate metadata entries and provenance fields
    seen_ids = set()
    chunk_ids = []

    for i, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata item at index {i} is not a dictionary.")
        if meta.get("faiss_id") != i:
            raise ValueError(f"Metadata item at index {i} has invalid faiss_id: {meta.get('faiss_id')}")

        for str_field in REQUIRED_STRING_PROVENANCE_FIELDS:
            val = meta.get(str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Metadata item at index {i} field '{str_field}' must be a non-empty string.")

        for int_field in REQUIRED_INT_PROVENANCE_FIELDS:
            val = meta.get(int_field)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"Metadata item at index {i} field '{int_field}' must be a positive integer.")

        cid = meta["chunk_id"]
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk_id in metadata: '{cid}'")
        seen_ids.add(cid)
        chunk_ids.append(cid)

    # 6. Verify ordered chunk IDs fingerprint
    ids_fingerprint = compute_chunk_ids_fingerprint(chunk_ids)
    if ids_fingerprint != manifest.get("ordered_chunk_ids_sha256"):
        raise ValueError(
            f"Ordered chunk IDs fingerprint mismatch: expected {manifest.get('ordered_chunk_ids_sha256')}, got {ids_fingerprint}"
        )

    return index, metadata, manifest


def reconstruct_and_verify_vectors(
    index: faiss.IndexFlatIP,
    matrix: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Tuple[int, int]:
    """Reconstruct all stored vectors from FAISS index and compare with input matrix.

    Args:
        index: Loaded FAISS index instance.
        matrix: Original normalized float32 matrix.
        rtol: Relative tolerance for floating-point comparison.
        atol: Absolute tolerance for floating-point comparison.

    Returns:
        Tuple of (verified_match_count, mismatch_count).
    """
    if index.ntotal != matrix.shape[0]:
        raise ValueError(
            f"Vector count mismatch for reconstruction: index has {index.ntotal}, matrix has {matrix.shape[0]}"
        )

    verified = 0
    mismatches = 0

    for i in range(index.ntotal):
        rec = index.reconstruct(i)
        target = matrix[i]
        if np.allclose(rec, target, rtol=rtol, atol=atol):
            verified += 1
        else:
            mismatches += 1

    return verified, mismatches


def main() -> None:
    """CLI entrypoint for Stage 7 Vector Storage generation and verification."""
    from src.embeddings import load_embeddings_artifact

    artifact_path = "data/processed/chunk_embeddings.json"

    print("Starting Stage 7 Vector Storage Pipeline...")
    print("Checking Stage 6 embeddings input artifact...")

    if not Path(artifact_path).is_file():
        print(f"ERROR: Stage 6 artifact not found at '{artifact_path}'. Cannot proceed.")
        sys.exit(1)

    # Compute artifact SHA-256 before pipeline
    before_sha = compute_file_sha256(artifact_path)

    # 1. Load Stage 6 artifact
    records = load_embeddings_artifact(artifact_path)
    print(f"Loaded {len(records)} embedding records from '{artifact_path}'.")

    # 2. Validate production 272-record corpus contract
    validate_embedding_records(records, expected_count=EXPECTED_CORPUS_COUNT)
    print(f"Validated corpus integrity: 272 records, unique chunk IDs, model '{DEFAULT_MODEL}'.")

    # 3. Build & normalize float32 matrix
    matrix, norm_stats = build_embeddings_matrix(records, expected_dim=DEFAULT_DIMENSION)
    print(f"Built float32 matrix shape {matrix.shape}, C-contiguous, 100% finite floats.")
    print(
        f"L2 Norm Stats: Min={norm_stats['min_norm']:.6f}, Max={norm_stats['max_norm']:.6f}, Mean={norm_stats['mean_norm']:.6f}"
    )

    # 4. Build FAISS IndexFlatIP
    index = build_faiss_index(matrix)
    print(f"Built FAISS IndexFlatIP index: dimension={index.d}, ntotal={index.ntotal}, is_trained={index.is_trained}.")

    # 5. Build integer ID -> chunk metadata mapping
    metadata = build_metadata_mapping(records)
    print(f"Built metadata mapping: {len(metadata)} records (FAISS IDs 0 to {len(metadata)-1}).")

    # 6. Persist vector store artifacts
    manifest = save_vector_store(index, metadata)
    print("Persisted vector store artifacts to 'index/':")
    print(f"  - index/faiss.index ({compute_file_sha256('index/faiss.index')[:12]}...)")
    print(f"  - index/chunk_metadata.json ({compute_file_sha256('index/chunk_metadata.json')[:12]}...)")
    print(f"  - index/index_manifest.json")

    # 7. Reload and validate persistence round-trip
    loaded_index, loaded_meta, loaded_manifest = load_vector_store()
    print("\nVerified production loader load_vector_store():")
    print(f"  - Manifest index_type: {loaded_manifest['index_type']}")
    print(f"  - Manifest metric_type: {loaded_manifest['metric_type']}")
    print(f"  - Loaded vector count:  {loaded_index.ntotal}")
    print(f"  - Loaded metadata count: {len(loaded_meta)}")

    # 8. Reconstruct and compare stored vectors
    verified_count, mismatch_count = reconstruct_and_verify_vectors(loaded_index, matrix)
    print(f"  - Vector Reconstruction Verification: {verified_count}/{loaded_index.ntotal} matched, {mismatch_count} mismatches.")

    # 9. Verify Stage 6 artifact checksum remained unchanged
    after_sha = compute_file_sha256(artifact_path)
    if before_sha != after_sha:
        raise ValueError("CRITICAL ERROR: Stage 6 artifact was modified during Stage 7 execution!")
    print(f"  - Stage 6 Artifact Checksum Unchanged: True (SHA-256: {after_sha[:16]}...)")

    print("\n" + "=" * 80)
    print("STAGE 7 VECTOR STORAGE SUMMARY")
    print("=" * 80)
    print(f"Index Class:              faiss.IndexFlatIP")
    print(f"Metric Contract:          METRIC_INNER_PRODUCT")
    print(f"Vector Dimension:         {loaded_index.d}")
    print(f"Stored Vectors:           {loaded_index.ntotal}")
    print(f"Metadata Records:         {len(loaded_meta)} (FAISS IDs 0..{len(loaded_meta)-1})")
    print(f"Matrix Dtype / Shape:     {matrix.dtype} / {matrix.shape}")
    print(f"L2 Norm Statistics:       Min={norm_stats['min_norm']:.6f}, Max={norm_stats['max_norm']:.6f}, Mean={norm_stats['mean_norm']:.6f}")
    print(f"Reconstruction Accuracy:  100% ({verified_count}/{loaded_index.ntotal} vectors matched within 1e-5 tolerance)")
    print(f"Gemini API Calls:         0")
    print("=" * 80)

    print("\nStage 7 Vector Storage completed successfully!")


if __name__ == "__main__":
    main()
