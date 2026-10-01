"""Unit test suite for Stage 8 Retrieval System module (src.retrieval).

Tests query formatting, Top-K parameter validation, query matrix conversion & defensive L2 normalization,
FAISS search & metadata mapping, result contract verification, prerequisite ordering, and error handling.
All tests run 100% offline using synthetic vector stores and SDK mocks, requiring zero network calls or API keys.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
from google.genai import errors

from src.retrieval import (
    DEFAULT_TOP_K,
    QUERY_PREFIX,
    embed_query,
    format_cli_results,
    prepare_query_matrix,
    prepare_query_text,
    retrieve,
    search_by_vector,
    validate_top_k,
)
from src.vector_store import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    build_embeddings_matrix,
    build_faiss_index,
    build_metadata_mapping,
    save_vector_store,
)


def create_synthetic_store(num_records: int = 10, dimension: int = DEFAULT_DIMENSION):
    """Helper to create a small synthetic vector store and records for offline unit tests."""
    records = []
    np.random.seed(100)
    for i in range(num_records):
        raw_vec = np.random.randn(dimension).tolist()
        records.append(
            {
                "chunk_id": f"doc_test_p001_c{i+1:03d}",
                "document_id": f"doc_test_{i//3 + 1}",
                "filename": f"test_doc_{i//3 + 1}.pdf",
                "page_number": (i % 3) + 1,
                "chunk_index": i + 1,
                "text": f"This is synthetic test chunk text number {i+1} covering animal research.",
                "title": f"Test Document Title {i//3 + 1}",
                "publisher": "Test Publisher Org",
                "source_url": "https://example.org/test_doc",
                "estimated_token_count": 60,
                "embedding": raw_vec,
                "embedding_model": DEFAULT_MODEL,
                "embedding_dimension": dimension,
            }
        )

    matrix, _ = build_embeddings_matrix(records, expected_dim=dimension)
    index = build_faiss_index(matrix)
    metadata = build_metadata_mapping(records)
    return index, metadata, matrix, records


class TestRetrievalSystem(unittest.TestCase):
    """Offline unit test suite for src.retrieval."""

    def setUp(self):
        """Set up synthetic test fixtures."""
        self.num_records = 10
        self.dim = DEFAULT_DIMENSION
        self.index, self.metadata, self.matrix, self.records = create_synthetic_store(
            num_records=self.num_records, dimension=self.dim
        )

    def test_01_prepare_query_text_formatting(self):
        """Test that query formatting prepends exact prefix and preserves original question string."""
        question = "How do bald eagles hunt for fish in rivers?"
        prepared = prepare_query_text(question)
        expected = f"{QUERY_PREFIX}{question}"
        self.assertEqual(prepared, expected)
        self.assertIn(question, prepared)
        self.assertTrue(prepared.startswith("task: question answering | query: "))

    def test_02_prepare_query_text_invalid_inputs(self):
        """Test that missing, non-string, empty, or whitespace-only questions raise ValueError."""
        invalid_inputs = [None, 123, True, "", "   ", "\n\t  "]
        for bad_q in invalid_inputs:
            with self.assertRaises(ValueError) as ctx:
                prepare_query_text(bad_q)  # type: ignore
            self.assertIn("non-empty string", str(ctx.exception).lower())

    def test_03_validate_top_k_valid(self):
        """Test valid top_k parameters."""
        self.assertEqual(validate_top_k(1, max_available=10), 1)
        self.assertEqual(validate_top_k(4, max_available=10), 4)
        self.assertEqual(validate_top_k(10, max_available=10), 10)

    def test_04_validate_top_k_invalid(self):
        """Test rejection of boolean, non-integer, zero, negative, or out-of-range top_k."""
        invalid_k_values = [True, False, 3.5, "4", 0, -1, 11, 100]
        for bad_k in invalid_k_values:
            with self.assertRaises(ValueError):
                validate_top_k(bad_k, max_available=10)

    def test_05_prepare_query_matrix_properties(self):
        """Test query matrix creation: float32, C-contiguous, shape (1, 768), and normalized ~1."""
        raw_vec = np.random.randn(self.dim).tolist()
        matrix = prepare_query_matrix(raw_vec, expected_dim=self.dim)

        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.shape, (1, self.dim))
        self.assertTrue(matrix.flags["C_CONTIGUOUS"])
        self.assertTrue(np.all(np.isfinite(matrix)))

        norm = float(np.linalg.norm(matrix))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_06_prepare_query_matrix_zero_norm_rejected(self):
        """Test that zero or near-zero norm query vectors are rejected."""
        zero_vec = [0.0] * self.dim
        with self.assertRaises(ValueError) as ctx:
            prepare_query_matrix(zero_vec, expected_dim=self.dim)
        self.assertIn("zero or near-zero", str(ctx.exception).lower())

    def test_07_search_by_vector_exact_top_k(self):
        """Test vector search returns exactly K results with correct fields, types, and ranks."""
        query_vec = list(self.records[2]["embedding"])  # Use record #2 vector
        query_matrix = prepare_query_matrix(query_vec, expected_dim=self.dim)

        results = search_by_vector(self.index, self.metadata, query_matrix, top_k=4)

        self.assertEqual(len(results), 4)
        # Record #2 should rank #1 because it was created from record #2's vector
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["faiss_id"], 2)
        self.assertEqual(results[0]["chunk_id"], self.records[2]["chunk_id"])
        self.assertAlmostEqual(results[0]["similarity_score"], 1.0, places=4)

        # Check fields and types for all returned items
        for idx, res in enumerate(results):
            self.assertEqual(res["rank"], idx + 1)
            self.assertIsInstance(res["faiss_id"], int)
            self.assertIsInstance(res["similarity_score"], float)
            self.assertIn("chunk_id", res)
            self.assertIn("text", res)
            self.assertNotIn("embedding", res)  # Exclude vector arrays
            self.assertEqual(res["text"], self.metadata[res["faiss_id"]]["text"])

    def test_08_search_by_vector_non_increasing_scores(self):
        """Test that search results return similarity scores in non-increasing order."""
        query_vec = np.random.randn(self.dim).tolist()
        query_matrix = prepare_query_matrix(query_vec, expected_dim=self.dim)

        results = search_by_vector(self.index, self.metadata, query_matrix, top_k=5)

        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]["similarity_score"] + 1e-6,
                results[i + 1]["similarity_score"],
            )

    def test_09_embed_query_mock_sdk_response(self):
        """Test embed_query with mocked GenAI client and SDK response types."""
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(self.dim).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.return_value = mock_response

        prepared_q = prepare_query_text("What do humpback whales eat?")
        vec = embed_query(mock_client, prepared_q, model_name=DEFAULT_MODEL, dimension=self.dim)

        self.assertEqual(len(vec), self.dim)
        self.assertTrue(all(isinstance(x, float) and np.isfinite(x) for x in vec))

        # Check call arguments
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(call_kwargs["contents"], prepared_q)
        self.assertEqual(call_kwargs["config"].output_dimensionality, self.dim)

    def test_10_embed_query_mock_sdk_empty_or_multiple_response(self):
        """Test embed_query rejects empty or multiple embedding responses."""
        mock_client = MagicMock()

        # Case A: Empty embeddings
        mock_response_empty = MagicMock()
        mock_response_empty.embeddings = []
        mock_client.models.embed_content.return_value = mock_response_empty

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text")
        self.assertIn("missing 'embeddings'", str(ctx.exception).lower())

        # Case B: Multiple embeddings
        mock_response_multi = MagicMock()
        mock_response_multi.embeddings = [MagicMock(), MagicMock()]
        mock_client.models.embed_content.return_value = mock_response_multi

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text")
        self.assertIn("expected exactly 1", str(ctx.exception).lower())

    def test_11_prerequisite_validation_order(self):
        """Test high-level retrieve() validates query, index, and top_k BEFORE network/client calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save synthetic store to tmpdir
            save_vector_store(self.index, self.metadata, index_dir=tmpdir)

            mock_client = MagicMock()

            # 1. Invalid question raises ValueError before calling embed_content
            with self.assertRaises(ValueError):
                retrieve("", index_dir=tmpdir, client=mock_client)
            mock_client.models.embed_content.assert_not_called()

            # 2. Missing index directory raises FileNotFoundError before calling embed_content
            with self.assertRaises(FileNotFoundError):
                retrieve("valid question?", index_dir="/nonexistent/dir", client=mock_client)
            mock_client.models.embed_content.assert_not_called()

            # 3. Invalid top_k raises ValueError before calling embed_content
            with self.assertRaises(ValueError):
                retrieve("valid question?", index_dir=tmpdir, top_k=999, client=mock_client)
            mock_client.models.embed_content.assert_not_called()

    def test_12_full_retrieve_pipeline_mocked(self):
        """Test high-level retrieve() orchestration end-to-end with mocked client."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(self.index, self.metadata, index_dir=tmpdir)

            mock_client = MagicMock()
            mock_embedding = MagicMock()
            # Use vector close to record #0
            mock_embedding.values = list(self.records[0]["embedding"])
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.models.embed_content.return_value = mock_response

            results = retrieve(
                "How do bald eagles nest?",
                index_dir=tmpdir,
                top_k=4,
                client=mock_client,
            )

            self.assertEqual(len(results), 4)
            self.assertEqual(results[0]["chunk_id"], self.records[0]["chunk_id"])
            self.assertEqual(results[0]["rank"], 1)

    def test_13_format_cli_results(self):
        """Test formatting retrieval results for CLI display."""
        query_vec = list(self.records[0]["embedding"])
        query_matrix = prepare_query_matrix(query_vec, expected_dim=self.dim)
        results = search_by_vector(self.index, self.metadata, query_matrix, top_k=2)

        formatted = format_cli_results("Test Query?", results)
        self.assertIn("STAGE 8 SEMANTIC RETRIEVAL RESULTS", formatted)
        self.assertIn('Query: "Test Query?"', formatted)
        self.assertIn("Rank #1", formatted)
        self.assertIn("Rank #2", formatted)
        self.assertIn(self.records[0]["chunk_id"], formatted)

    def test_14_no_gemini_api_key_required_for_offline_tests(self):
        """Verify that offline search_by_vector and helper functions require zero network or API keys."""
        env_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            query_vec = list(self.records[0]["embedding"])
            matrix = prepare_query_matrix(query_vec, expected_dim=self.dim)
            results = search_by_vector(self.index, self.metadata, matrix, top_k=3)
            self.assertEqual(len(results), 3)
        finally:
            if env_key:
                os.environ["GEMINI_API_KEY"] = env_key

    @patch("src.retrieval.time.sleep")
    def test_15_retry_behavior_401(self, mock_sleep):
        """Prove 401 error results in exactly one attempt, zero sleep, and immediate failure."""
        mock_client = MagicMock()
        err_401 = errors.APIError(401, {"error": {"message": "Unauthorized API key", "code": 401}})
        mock_client.models.embed_content.side_effect = err_401

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertIn("non-transient", str(ctx.exception).lower())

    @patch("src.retrieval.time.sleep")
    def test_16_retry_behavior_403(self, mock_sleep):
        """Prove 403 error results in exactly one attempt, zero sleep, and immediate failure."""
        mock_client = MagicMock()
        err_403 = errors.APIError(403, {"error": {"message": "Forbidden access", "code": 403}})
        mock_client.models.embed_content.side_effect = err_403

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertIn("non-transient", str(ctx.exception).lower())

    @patch("src.retrieval.time.sleep")
    def test_17_retry_behavior_503_then_success(self, mock_sleep):
        """Prove 503 followed by success results in two attempts and one bounded sleep."""
        mock_client = MagicMock()
        err_503 = errors.APIError(503, {"error": {"message": "Service Unavailable", "code": 503}})

        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(self.dim).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.side_effect = [err_503, mock_response]

        vec = embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(len(vec), self.dim)
        self.assertEqual(mock_client.models.embed_content.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("src.retrieval.time.sleep")
    def test_18_retry_behavior_repeated_transient_failures(self, mock_sleep):
        """Prove max_retries + 1 attempts on repeated transient errors, then clear failure."""
        mock_client = MagicMock()
        err_503 = errors.APIError(503, {"error": {"message": "Service Unavailable", "code": 503}})
        mock_client.models.embed_content.side_effect = err_503

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(mock_client.models.embed_content.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_has_calls([unittest.mock.call(1.0), unittest.mock.call(2.0), unittest.mock.call(4.0)])
        self.assertIn("exhausted retries", str(ctx.exception).lower())

    @patch("src.retrieval.time.sleep")
    def test_19_retry_behavior_max_retries_zero(self, mock_sleep):
        """Prove max_retries=0 allows exactly 1 attempt and no sleep on transient failure."""
        mock_client = MagicMock()
        err_503 = errors.APIError(503, {"error": {"message": "Service Unavailable", "code": 503}})
        mock_client.models.embed_content.side_effect = err_503

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text", max_retries=0, retry_delay=1.0)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertIn("exhausted retries", str(ctx.exception).lower())

    @patch("src.retrieval.time.sleep")
    def test_20_retry_behavior_429_bounded(self, mock_sleep):
        """Prove retryable 429 rate limit is retried with bounded wait and eventual success."""
        mock_client = MagicMock()
        err_429 = errors.APIError(429, {"error": {"message": "Resource Exhausted", "code": 429}})

        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(self.dim).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.side_effect = [err_429, mock_response]

        vec = embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(len(vec), self.dim)
        self.assertEqual(mock_client.models.embed_content.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("src.retrieval.time.sleep")
    def test_20b_daily_quota_429_is_not_retried(self, mock_sleep):
        """Prove an exhausted daily quota fails after one call without sleeping."""
        mock_client = MagicMock()
        daily_quota = errors.APIError(
            429,
            {
                "error": {
                    "message": (
                        "Quota exceeded for limit "
                        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
                    ),
                    "code": 429,
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )
        mock_client.models.embed_content.side_effect = daily_quota

        with self.assertRaises(ValueError) as ctx:
            embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertIn("daily quota", str(ctx.exception).lower())

    @patch("src.retrieval.time.sleep")
    def test_21_retry_behavior_success_first_attempt(self, mock_sleep):
        """Prove successful first attempt causes exactly 1 API call and zero sleeps."""
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = np.random.randn(self.dim).tolist()
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_client.models.embed_content.return_value = mock_response

        vec = embed_query(mock_client, "query text", max_retries=3, retry_delay=1.0)

        self.assertEqual(len(vec), self.dim)
        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        mock_sleep.assert_not_called()

    def test_22_model_mismatch_rejected_before_api_call(self):
        """Prove model mismatch with same 768 dimensions is rejected BEFORE any API call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(self.index, self.metadata, index_dir=tmpdir)
            mock_client = MagicMock()

            with self.assertRaises(ValueError) as ctx:
                retrieve(
                    "What do bald eagles eat?",
                    index_dir=tmpdir,
                    model_name="gemini-embedding-1",
                    dimension=DEFAULT_DIMENSION,
                    client=mock_client,
                )

            mock_client.models.embed_content.assert_not_called()
            self.assertIn("incompatible with index manifest model", str(ctx.exception).lower())

    def test_23_dimension_mismatch_rejected_before_api_call(self):
        """Prove dimension mismatch is rejected BEFORE any API call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(self.index, self.metadata, index_dir=tmpdir)
            mock_client = MagicMock()

            with self.assertRaises(ValueError) as ctx:
                retrieve(
                    "What do bald eagles eat?",
                    index_dir=tmpdir,
                    model_name=DEFAULT_MODEL,
                    dimension=512,
                    client=mock_client,
                )

            mock_client.models.embed_content.assert_not_called()
            self.assertIn("incompatible with index manifest dimension", str(ctx.exception).lower())

    def test_24_matching_model_and_dimension_proceed_normally(self):
        """Prove matching model and dimension settings proceed normally through embedding & retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_vector_store(self.index, self.metadata, index_dir=tmpdir)
            mock_client = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.values = list(self.records[0]["embedding"])
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.models.embed_content.return_value = mock_response

            results = retrieve(
                "What do bald eagles eat?",
                index_dir=tmpdir,
                model_name=DEFAULT_MODEL,
                dimension=DEFAULT_DIMENSION,
                client=mock_client,
            )

            mock_client.models.embed_content.assert_called_once()
            self.assertEqual(len(results), DEFAULT_TOP_K)
            self.assertEqual(results[0]["chunk_id"], self.records[0]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
