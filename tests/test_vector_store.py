"""Unit test suite for Stage 7 Vector Storage module (src.vector_store).

Tests FAISS IndexFlatIP construction, float32 matrix conversion, defensive L2-normalization,
metadata mapping, atomic persistence, validating artifact loader, and vector reconstruction.
All tests run 100% offline using synthetic vector fixtures and temporary directories.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from src.vector_store import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    build_embeddings_matrix,
    build_faiss_index,
    build_metadata_mapping,
    compute_chunk_ids_fingerprint,
    compute_file_sha256,
    load_vector_store,
    reconstruct_and_verify_vectors,
    save_vector_store,
    validate_embedding_records,
)


def create_synthetic_embedding_records(
    num_records: int = 5,
    dimension: int = DEFAULT_DIMENSION,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """Helper to create distinct synthetic embedding records for offline testing.

    Uses distinct orthogonal or non-identical floating point vectors to enable
    detection of swapped or reordered mappings.
    """
    records = []
    np.random.seed(42)  # Deterministic seed for reproducible tests
    for i in range(num_records):
        # Create distinct non-zero vector
        raw_vec = np.random.randn(dimension).tolist()
        records.append(
            {
                "chunk_id": f"doc_test_p001_c{i+1:03d}",
                "document_id": "doc_test",
                "filename": "test_doc.pdf",
                "page_number": 1,
                "chunk_index": i + 1,
                "text": f"This is synthetic test passage {i+1} about animal biology.",
                "title": "Test Animal Document",
                "publisher": "Test Publisher",
                "source_url": "https://example.org/test",
                "estimated_token_count": 50,
                "embedding": raw_vec,
                "embedding_model": model_name,
                "embedding_dimension": dimension,
            }
        )
    return records


class TestVectorStore(unittest.TestCase):
    """Test suite for src.vector_store."""

    def setUp(self):
        """Set up synthetic test fixtures."""
        self.num_records = 4
        self.dim = DEFAULT_DIMENSION
        self.records = create_synthetic_embedding_records(
            num_records=self.num_records, dimension=self.dim
        )

    def test_01_validate_embedding_records_valid(self):
        """Test that valid embedding records pass validation without error."""
        validate_embedding_records(self.records, expected_count=self.num_records, expected_dim=self.dim)

    def test_02_validate_embedding_records_wrong_count(self):
        """Test that validation fails when record count does not match expected_count."""
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_records(self.records, expected_count=999)
        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_03_validate_embedding_records_duplicate_chunk_id(self):
        """Test that duplicate chunk IDs in record collection are rejected."""
        bad_records = [dict(r) for r in self.records]
        bad_records[1]["chunk_id"] = bad_records[0]["chunk_id"]
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_records(bad_records)
        self.assertIn("duplicate chunk_id", str(ctx.exception).lower())

    def test_04_validate_embedding_records_wrong_model(self):
        """Test that non-matching embedding model name is rejected."""
        bad_records = [dict(r) for r in self.records]
        bad_records[0]["embedding_model"] = "wrong-model-v1"
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_records(bad_records, expected_model=DEFAULT_MODEL)
        self.assertIn("model mismatch", str(ctx.exception).lower())

    def test_05_validate_embedding_records_wrong_dimension(self):
        """Test that vectors with wrong dimensions are rejected."""
        bad_records = [dict(r) for r in self.records]
        bad_records[0]["embedding"] = [0.1] * 100  # Wrong dimension
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_records(bad_records, expected_dim=DEFAULT_DIMENSION)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_06_validate_embedding_records_non_finite(self):
        """Test that non-finite values (NaN / Infinity) are rejected."""
        bad_records = [dict(r) for r in self.records]
        bad_records[0]["embedding"] = list(bad_records[0]["embedding"])
        bad_records[0]["embedding"][5] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_records(bad_records)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_07_build_embeddings_matrix_properties(self):
        """Test that matrix construction creates a C-contiguous float32 array with normalized row norms ~1."""
        matrix, norm_stats = build_embeddings_matrix(self.records, expected_dim=self.dim)
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (self.num_records, self.dim))
        self.assertTrue(matrix.flags["C_CONTIGUOUS"])
        self.assertTrue(np.all(np.isfinite(matrix)))

        # Verify post-normalization row norms
        norms = np.linalg.norm(matrix, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))
        self.assertAlmostEqual(norm_stats["min_norm"], 1.0, places=4)
        self.assertAlmostEqual(norm_stats["max_norm"], 1.0, places=4)

    def test_08_build_embeddings_matrix_zero_norm_rejected(self):
        """Test that zero or near-zero norm vectors are rejected."""
        bad_records = [dict(r) for r in self.records]
        bad_records[0]["embedding"] = [0.0] * self.dim
        with self.assertRaises(ValueError) as ctx:
            build_embeddings_matrix(bad_records)
        self.assertIn("zero or near-zero norm", str(ctx.exception).lower())

    def test_09_original_records_not_mutated(self):
        """Test that building matrix and metadata does not mutate input record dictionaries."""
        orig_copy = json.dumps(self.records)
        build_embeddings_matrix(self.records)
        build_metadata_mapping(self.records)
        self.assertEqual(json.dumps(self.records), orig_copy)

    def test_10_build_faiss_index_properties(self):
        """Test that build_faiss_index creates a trained IndexFlatIP index with correct metric and count."""
        matrix, _ = build_embeddings_matrix(self.records, expected_dim=self.dim)
        index = build_faiss_index(matrix)

        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(index.d, self.dim)
        self.assertEqual(index.ntotal, self.num_records)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)

    def test_11_build_metadata_mapping(self):
        """Test metadata mapping generation, 0-indexed faiss_id, and exclusion of full embedding vector."""
        metadata = build_metadata_mapping(self.records)
        self.assertEqual(len(metadata), self.num_records)

        for i, meta in enumerate(metadata):
            self.assertEqual(meta["faiss_id"], i)
            self.assertEqual(meta["chunk_id"], self.records[i]["chunk_id"])
            self.assertEqual(meta["text"], self.records[i]["text"])
            self.assertNotIn("embedding", meta)  # Full embedding vector excluded

    def test_12_save_and_load_vector_store_round_trip(self):
        """Test full save and load round-trip in a temporary directory."""
        matrix, _ = build_embeddings_matrix(self.records)
        index = build_faiss_index(matrix)
        metadata = build_metadata_mapping(self.records)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(index, metadata, index_dir=tmpdir)

            # Check that files exist
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "faiss.index")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "chunk_metadata.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "index_manifest.json")))

            # Load back and validate
            loaded_idx, loaded_meta, manifest = load_vector_store(index_dir=tmpdir)

            self.assertEqual(loaded_idx.ntotal, self.num_records)
            self.assertEqual(loaded_idx.d, self.dim)
            self.assertEqual(len(loaded_meta), self.num_records)
            self.assertEqual(manifest["index_type"], "IndexFlatIP")
            self.assertEqual(manifest["metric_type"], "METRIC_INNER_PRODUCT")

            # Verify vector reconstruction
            verified, mismatches = reconstruct_and_verify_vectors(loaded_idx, matrix)
            self.assertEqual(verified, self.num_records)
            self.assertEqual(mismatches, 0)

    def test_13_load_vector_store_missing_file(self):
        """Test that loader raises FileNotFoundError if any required file is missing."""
        matrix, _ = build_embeddings_matrix(self.records)
        index = build_faiss_index(matrix)
        metadata = build_metadata_mapping(self.records)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(index, metadata, index_dir=tmpdir)
            # Remove index file
            os.remove(os.path.join(tmpdir, "faiss.index"))

            with self.assertRaises(FileNotFoundError):
                load_vector_store(index_dir=tmpdir)

    def test_14_load_vector_store_checksum_mismatch(self):
        """Test that loader detects corrupted file content via checksum mismatch."""
        matrix, _ = build_embeddings_matrix(self.records)
        index = build_faiss_index(matrix)
        metadata = build_metadata_mapping(self.records)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(index, metadata, index_dir=tmpdir)

            # Corrupt chunk_metadata.json
            meta_path = os.path.join(tmpdir, "chunk_metadata.json")
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write("\n/* corrupted */")

            with self.assertRaises(ValueError) as ctx:
                load_vector_store(index_dir=tmpdir)
            self.assertIn("checksum mismatch", str(ctx.exception).lower())

    def test_15_load_vector_store_reordered_metadata(self):
        """Test that loader detects reordered metadata via ordered ID fingerprint mismatch."""
        matrix, _ = build_embeddings_matrix(self.records)
        index = build_faiss_index(matrix)
        metadata = build_metadata_mapping(self.records)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(index, metadata, index_dir=tmpdir)

            # Swap metadata items 0 and 1
            meta_path = os.path.join(tmpdir, "chunk_metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data[0], data[1] = data[1], data[0]
            # Fix faiss_id to bypass basic ID check and test fingerprint check
            data[0]["faiss_id"] = 0
            data[1]["faiss_id"] = 1

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            # Re-compute checksum in manifest to isolate the fingerprint check
            manifest_path = os.path.join(tmpdir, "index_manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            man["chunk_metadata_sha256"] = compute_file_sha256(meta_path)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(man, f)

            with self.assertRaises(ValueError) as ctx:
                load_vector_store(index_dir=tmpdir)
            self.assertIn("fingerprint mismatch", str(ctx.exception).lower())

    def test_16_no_gemini_api_key_required(self):
        """Verify that vector_store functions operate 100% offline without GEMINI_API_KEY."""
        # Save current env key if present
        env_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            matrix, _ = build_embeddings_matrix(self.records)
            index = build_faiss_index(matrix)
            metadata = build_metadata_mapping(self.records)
            with tempfile.TemporaryDirectory() as tmpdir:
                save_vector_store(index, metadata, index_dir=tmpdir)
                loaded_idx, loaded_meta, _ = load_vector_store(index_dir=tmpdir)
                self.assertEqual(loaded_idx.ntotal, self.num_records)
        finally:
            if env_key:
                os.environ["GEMINI_API_KEY"] = env_key


if __name__ == "__main__":
    unittest.main()
