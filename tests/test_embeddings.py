"""Unit tests for Stage 6 embeddings module."""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from google.genai import types

from src.embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    calculate_l2_norm,
    create_genai_client,
    embed_corpus_chunks,
    embed_single_chunk,
    load_embeddings_artifact,
    prepare_document_text,
    save_embeddings_artifact,
    validate_embedding_vector,
)


class TestEmbeddings(unittest.TestCase):
    """Offline unit tests for Stage 6 embedding module using SDK types and mocks."""

    def setUp(self):
        self.sample_chunk = {
            "chunk_id": "doc_bald_eagle_p001_c001",
            "document_id": "doc_bald_eagle",
            "filename": "bald_eagle_factsheet.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Bald eagle biology and nesting habits text.",
            "title": "U.S. Fish & Wildlife Service Species Fact Sheet: Bald Eagle",
            "publisher": "U.S. Fish and Wildlife Service",
            "source_url": "https://www.fws.gov/bald-eagle-fact-sheet.pdf",
            "estimated_token_count": 10,
        }
        # Create a mock 768-dim vector normalized to length ~1.0
        val = 1.0 / math.sqrt(768)
        self.valid_vector = [val] * 768

    def _make_sdk_response(self, vector):
        emb = types.ContentEmbedding(values=vector)
        return types.EmbedContentResponse(embeddings=[emb])

    def test_01_prepare_document_text_formatting(self):
        formatted = prepare_document_text(self.sample_chunk)
        expected = (
            "title: U.S. Fish & Wildlife Service Species Fact Sheet: Bald Eagle | "
            "text: Bald eagle biology and nesting habits text."
        )
        self.assertEqual(formatted, expected)

    def test_02_prepare_document_text_rejects_empty(self):
        empty_chunk = dict(self.sample_chunk, text="")
        with self.assertRaises(ValueError):
            prepare_document_text(empty_chunk)

    def test_03_no_unrelated_metadata_in_formatted_text(self):
        formatted = prepare_document_text(self.sample_chunk)
        self.assertNotIn("bald_eagle_factsheet.pdf", formatted)
        self.assertNotIn("doc_bald_eagle_p001_c001", formatted)
        self.assertNotIn("https://www.fws.gov", formatted)

    def test_04_create_genai_client_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                create_genai_client(api_key=None)
            self.assertIn("GEMINI_API_KEY is not set", str(ctx.exception))

    def test_05_validate_embedding_vector_success(self):
        validated = validate_embedding_vector(self.valid_vector, expected_dim=768)
        self.assertEqual(len(validated), 768)
        self.assertAlmostEqual(calculate_l2_norm(validated), 1.0, places=4)

    def test_06_validate_embedding_vector_rejects_wrong_dim(self):
        with self.assertRaises(ValueError) as ctx:
            validate_embedding_vector([0.1] * 128, expected_dim=768)
        self.assertIn("dimension mismatch", str(ctx.exception))

    def test_07_validate_embedding_vector_rejects_non_numeric(self):
        invalid_vec = list(self.valid_vector)
        invalid_vec[0] = "string_value"
        with self.assertRaises(ValueError):
            validate_embedding_vector(invalid_vec)

    def test_08_validate_embedding_vector_rejects_boolean(self):
        invalid_vec = list(self.valid_vector)
        invalid_vec[0] = True
        with self.assertRaises(ValueError):
            validate_embedding_vector(invalid_vec)

    def test_09_validate_embedding_vector_rejects_nan_and_inf(self):
        vec_nan = list(self.valid_vector)
        vec_nan[0] = float("nan")
        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_nan)

        vec_inf = list(self.valid_vector)
        vec_inf[0] = float("inf")
        with self.assertRaises(ValueError):
            validate_embedding_vector(vec_inf)

    def test_10_embed_single_chunk_sdk_response_success(self):
        mock_client = MagicMock()
        mock_response = self._make_sdk_response(self.valid_vector)
        mock_client.models.embed_content.return_value = mock_response

        record = embed_single_chunk(mock_client, self.sample_chunk)

        # Check call arguments
        mock_client.models.embed_content.assert_called_once()
        _, kwargs = mock_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(kwargs["config"].output_dimensionality, DEFAULT_DIMENSION)

        # Check returned record fields
        self.assertEqual(record["chunk_id"], self.sample_chunk["chunk_id"])
        self.assertEqual(record["embedding"], self.valid_vector)
        self.assertEqual(record["embedding_model"], DEFAULT_MODEL)
        self.assertEqual(record["embedding_dimension"], DEFAULT_DIMENSION)
        self.assertEqual(record["title"], self.sample_chunk["title"])

    def test_11_zero_embeddings_rejected(self):
        mock_client = MagicMock()
        mock_response = types.EmbedContentResponse(embeddings=[])
        mock_client.models.embed_content.return_value = mock_response

        with patch("time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                embed_single_chunk(mock_client, self.sample_chunk)
        self.assertIn("missing 'embeddings' field or embeddings list is empty", str(ctx.exception))

    def test_12_multiple_embeddings_rejected(self):
        mock_client = MagicMock()
        emb1 = types.ContentEmbedding(values=self.valid_vector)
        emb2 = types.ContentEmbedding(values=self.valid_vector)
        mock_response = types.EmbedContentResponse(embeddings=[emb1, emb2])
        mock_client.models.embed_content.return_value = mock_response

        with patch("time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                embed_single_chunk(mock_client, self.sample_chunk)
        self.assertIn("expected exactly 1", str(ctx.exception))

    def test_13_missing_vector_values_rejected(self):
        mock_client = MagicMock()
        emb = types.ContentEmbedding(values=None)
        mock_response = types.EmbedContentResponse(embeddings=[emb])
        mock_client.models.embed_content.return_value = mock_response

        with patch("time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                embed_single_chunk(mock_client, self.sample_chunk)
        self.assertIn("missing 'values' field", str(ctx.exception))

    def test_14_input_records_not_mutated(self):
        mock_client = MagicMock()
        mock_response = self._make_sdk_response(self.valid_vector)
        mock_client.models.embed_content.return_value = mock_response

        chunk_copy = dict(self.sample_chunk)
        record = embed_single_chunk(mock_client, chunk_copy)

        self.assertNotIn("embedding", self.sample_chunk)
        self.assertNotIn("embedding", chunk_copy)
        self.assertIn("embedding", record)

    def test_15_non_transient_auth_error_fails_immediately(self):
        mock_client = MagicMock()
        mock_client.models.embed_content.side_effect = Exception("API_KEY_INVALID: 401 Unauthorized")

        with patch("time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                embed_single_chunk(mock_client, self.sample_chunk, max_retries=3)

        self.assertEqual(mock_client.models.embed_content.call_count, 1)
        self.assertIn("authentication or permission error", str(ctx.exception))

    def test_16_transient_error_retries_and_recovers(self):
        mock_client = MagicMock()
        mock_response = self._make_sdk_response(self.valid_vector)

        # Fail on attempt 1, succeed on attempt 2
        mock_client.models.embed_content.side_effect = [
            Exception("503 Service Unavailable"),
            mock_response,
        ]

        with patch("time.sleep") as mock_sleep:
            record = embed_single_chunk(mock_client, self.sample_chunk, max_retries=3)

        self.assertEqual(mock_client.models.embed_content.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)
        self.assertEqual(record["embedding"], self.valid_vector)

    def test_17_duplicate_input_ids_rejected_before_network(self):
        chunks = [dict(self.sample_chunk), dict(self.sample_chunk)]
        mock_client = MagicMock()

        with self.assertRaises(ValueError) as ctx:
            embed_corpus_chunks(chunks, client=mock_client)

        self.assertIn("Duplicate chunk_id", str(ctx.exception))
        mock_client.models.embed_content.assert_not_called()

    def test_18_corpus_embedding_preserves_order_and_mapping(self):
        chunk1 = dict(self.sample_chunk, chunk_id="doc_a_p001_c001", text="Chunk A text")
        chunk2 = dict(self.sample_chunk, chunk_id="doc_b_p001_c001", text="Chunk B text")

        vec1 = [0.1] * 768
        vec2 = [0.2] * 768

        mock_client = MagicMock()
        mock_resp1 = self._make_sdk_response(vec1)
        mock_resp2 = self._make_sdk_response(vec2)

        mock_client.models.embed_content.side_effect = [mock_resp1, mock_resp2]

        results = embed_corpus_chunks([chunk1, chunk2], client=mock_client, delay_between_requests=0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "doc_a_p001_c001")
        self.assertEqual(results[0]["embedding"], vec1)
        self.assertEqual(results[1]["chunk_id"], "doc_b_p001_c001")
        self.assertEqual(results[1]["embedding"], vec2)

    def test_19_save_and_load_artifact(self):
        record = dict(self.sample_chunk, embedding=self.valid_vector, embedding_model=DEFAULT_MODEL, embedding_dimension=768)
        with tempfile.TemporaryDirectory() as tmp_dir:
            art_path = os.path.join(tmp_dir, "test_embeddings.json")
            save_embeddings_artifact([record], art_path)
            self.assertTrue(os.path.isfile(art_path))

            loaded = load_embeddings_artifact(art_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["chunk_id"], self.sample_chunk["chunk_id"])
            self.assertEqual(loaded[0]["embedding"], self.valid_vector)


if __name__ == "__main__":
    unittest.main()
