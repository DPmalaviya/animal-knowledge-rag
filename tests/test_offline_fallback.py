"""Tests for deterministic, network-free corpus retrieval."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.generation import build_source_map
from src.offline_fallback import rank_metadata, retrieve_locally, tokenize_meaningful


def make_record(faiss_id, *, title="Animal research", text="Animal evidence"):
    """Return one valid chunk-metadata record."""
    return {
        "faiss_id": faiss_id,
        "chunk_id": f"document_p001_c{faiss_id + 1:03d}",
        "document_id": f"document-{faiss_id}",
        "filename": f"document-{faiss_id}.pdf",
        "page_number": 1,
        "chunk_index": faiss_id + 1,
        "text": text,
        "title": title,
        "publisher": "Example Publisher",
        "source_url": "https://example.org/document",
        "estimated_token_count": 20,
        "embedding_model": "gemini-embedding-2",
        "embedding_dimension": 768,
    }


def write_artifacts(index_dir, records, manifest_overrides=None):
    """Write metadata plus a matching minimal integrity manifest."""
    metadata_path = Path(index_dir) / "chunk_metadata.json"
    metadata_path.write_text(json.dumps(records), encoding="utf-8")
    chunk_ids = [record["chunk_id"] for record in records]
    manifest = {
        "metadata_count": len(records),
        "embedding_model": "gemini-embedding-2",
        "dimension": 768,
        "chunk_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "ordered_chunk_ids_sha256": hashlib.sha256(
            json.dumps(chunk_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    manifest.update(manifest_overrides or {})
    (Path(index_dir) / "index_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return metadata_path


class TestLocalLexicalRetrieval(unittest.TestCase):
    def test_tokenization_is_lowercase_alphanumeric(self):
        self.assertEqual(
            tokenize_meaningful("Bald-EAGLE's Pb2 exposure in 2024!"),
            ["bald", "eagle", "pb2", "exposure", "2024"],
        )

    def test_tokenization_removes_stop_words_and_short_non_numeric_tokens(self):
        self.assertEqual(
            tokenize_meaningful("The ox is at a zoo with 2 emus and an owl"),
            ["zoo", "2", "emus", "owl"],
        )

    def test_title_matches_are_weighted_three_times(self):
        metadata = [
            make_record(0, title="eagle", text="other words here"),
            make_record(1, title="other words", text="eagle eagle"),
        ]
        results = rank_metadata("eagle", metadata, top_k=2)
        self.assertEqual([record["faiss_id"] for record in results], [0, 1])

    def test_score_formula_uses_term_frequency_and_candidate_length(self):
        metadata = [make_record(0, title="eagle eagle", text="eagle wing flight")]
        result = rank_metadata("eagle", metadata, top_k=1)[0]
        # numerator = (2 title occurrences * 3) + 1 text occurrence;
        # denominator = sqrt(2 title tokens + 3 text tokens).
        self.assertAlmostEqual(result["similarity_score"], 7 / math.sqrt(5))

    def test_repeated_query_terms_do_not_multiply_the_score(self):
        metadata = [make_record(0, title="eagle", text="eagle flight")]
        single = rank_metadata("eagle", metadata, top_k=1)[0]["similarity_score"]
        repeated = rank_metadata("eagle eagle", metadata, top_k=1)[0]["similarity_score"]
        self.assertEqual(single, repeated)

    def test_stable_ties_sort_by_ascending_faiss_id(self):
        metadata = [
            make_record(0, title="eagle", text="wing flight"),
            make_record(1, title="eagle", text="nest flight"),
        ]
        results = rank_metadata("eagle", metadata, top_k=2)
        self.assertEqual([record["faiss_id"] for record in results], [0, 1])

    def test_top_k_limits_results_and_assigns_contiguous_ranks(self):
        metadata = [make_record(i, text="eagle evidence") for i in range(3)]
        results = rank_metadata("eagle", metadata, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([record["rank"] for record in results], [1, 2])

    def test_records_without_meaningful_query_overlap_are_omitted(self):
        metadata = [make_record(0, text="whale migration")]
        self.assertEqual(rank_metadata("eagle lead", metadata, top_k=1), [])
        self.assertEqual(rank_metadata("the is an", metadata, top_k=1), [])

    def test_results_satisfy_generation_contract(self):
        results = rank_metadata(
            "eagle", [make_record(0, title="Eagle research")], top_k=1
        )
        source_map = build_source_map(results)
        self.assertEqual(source_map["C1"]["faiss_id"], 0)
        self.assertTrue(math.isfinite(results[0]["similarity_score"]))

    def test_ranking_returns_copies_without_mutating_metadata(self):
        metadata = [make_record(0, title="Eagle research")]
        original = copy.deepcopy(metadata)
        results = rank_metadata("eagle", metadata, top_k=1)
        results[0]["text"] = "changed"
        self.assertEqual(metadata, original)

    def test_invalid_question_and_top_k_are_rejected(self):
        metadata = [make_record(0)]
        for question in (None, "", "   ", 3):
            with self.subTest(question=question), self.assertRaises(ValueError):
                rank_metadata(question, metadata, top_k=1)
        for top_k in (True, False, 0, -1, 1.5, "1", None):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                rank_metadata("animal", metadata, top_k=top_k)
        with self.assertRaises(ValueError):
            rank_metadata("animal", metadata, top_k=2)

    def test_retrieve_rejects_missing_and_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                retrieve_locally("eagle", temp_dir, top_k=1)

            metadata_path = Path(temp_dir) / "chunk_metadata.json"
            metadata_path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                retrieve_locally("eagle", temp_dir, top_k=1)

            for value in ({}, []):
                metadata_path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(value=value), self.assertRaises(ValueError):
                    retrieve_locally("eagle", temp_dir, top_k=1)

    def test_retrieve_rejects_missing_or_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "chunk_metadata.json"
            metadata_path.write_text(json.dumps([make_record(0)]), encoding="utf-8")
            with self.assertRaises(ValueError):
                retrieve_locally("animal", temp_dir, top_k=1)

            manifest_path = Path(temp_dir) / "index_manifest.json"
            for manifest_text in ("{not json", "[]"):
                manifest_path.write_text(manifest_text, encoding="utf-8")
                with self.subTest(manifest_text=manifest_text), self.assertRaises(ValueError):
                    retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_manifest_contract_mismatches(self):
        mismatches = (
            {"metadata_count": 2},
            {"embedding_model": "other-model"},
            {"dimension": 512},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for override in mismatches:
                write_artifacts(temp_dir, [make_record(0)], override)
                with self.subTest(override=override), self.assertRaises(ValueError):
                    retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_metadata_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = write_artifacts(temp_dir, [make_record(0)])
            metadata_path.write_text(
                json.dumps([make_record(0, text="tampered animal evidence")]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_ordered_chunk_id_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_artifacts(
                temp_dir,
                [make_record(0)],
                {"ordered_chunk_ids_sha256": "0" * 64},
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_malformed_metadata_records(self):
        malformed_cases = []
        missing = make_record(0)
        del missing["title"]
        malformed_cases.append([missing])
        bad_type = make_record(0)
        bad_type["page_number"] = True
        malformed_cases.append([bad_type])
        bad_model = make_record(0)
        bad_model["embedding_model"] = "other-model"
        malformed_cases.append([bad_model])

        with tempfile.TemporaryDirectory() as temp_dir:
            for records in malformed_cases:
                write_artifacts(temp_dir, records)
                with self.subTest(records=records), self.assertRaises(ValueError):
                    retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_duplicate_chunk_ids(self):
        records = [make_record(0), make_record(1)]
        records[1]["chunk_id"] = records[0]["chunk_id"]
        with tempfile.TemporaryDirectory() as temp_dir:
            write_artifacts(temp_dir, records)
            with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
                retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_duplicate_or_out_of_order_faiss_ids(self):
        cases = [
            [make_record(0), make_record(0)],
            [make_record(1), make_record(0)],
            [make_record(0), make_record(2)],
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for records in cases:
                write_artifacts(temp_dir, records)
                with self.subTest(ids=[r["faiss_id"] for r in records]):
                    with self.assertRaises(ValueError):
                        retrieve_locally("animal", temp_dir, top_k=1)

    def test_retrieve_rejects_top_k_above_available_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_artifacts(temp_dir, [make_record(0)])
            with self.assertRaises(ValueError):
                retrieve_locally("animal", temp_dir, top_k=2)

    def test_real_artifact_smoke_query_bald_eagle_lead_exposure(self):
        results = retrieve_locally(
            "How are bald eagles exposed to lead?", index_dir="index", top_k=4
        )
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 4)
        self.assertIn("bald eagle", results[0]["title"].lower())
        self.assertIn("lead", (results[0]["title"] + results[0]["text"]).lower())
        build_source_map(results)


if __name__ == "__main__":
    unittest.main()
