"""Tests for deterministic, network-free corpus retrieval."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.generation import (
    DEFAULT_GENERATION_MODEL,
    INSUFFICIENT_CONTEXT_FALLBACK,
    build_source_map,
)
from src.offline_fallback import (
    answer_with_free_fallback,
    build_extractive_rag_result,
    build_offline_fallback_result,
    rank_metadata,
    resolve_extractive_result,
    retrieve_locally,
    split_sentences,
    tokenize_meaningful,
)


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


class TestExtractiveAnswer(unittest.TestCase):
    def ranked_record(self, faiss_id, *, text, rank=1, page_number=1):
        record = make_record(faiss_id, text=text)
        record.update(
            rank=rank,
            similarity_score=1.0,
            page_number=page_number,
        )
        return record

    def padded_sentence(self, prefix, length):
        return prefix + ("x" * (length - len(prefix) - 1)) + "."

    def test_sentence_splitting_uses_terminal_punctuation_and_whitespace(self):
        text = "  First exact sentence!\nSecond one?  Third. Fourth fragment  "
        self.assertEqual(
            split_sentences(text),
            ["First exact sentence!", "Second one?", "Third.", "Fourth fragment"],
        )

    def test_scores_frequency_then_breaks_ties_by_rank_and_position(self):
        records = [
            self.ranked_record(
                0,
                rank=1,
                text="Eagle appears once. Eagle appears again.",
            ),
            self.ranked_record(
                1,
                rank=2,
                text="Eagle eagle has greater frequency. Eagle is later.",
            ),
        ]
        result = build_extractive_rag_result("eagle", records)
        self.assertEqual(
            result["answer"],
            "Eagle eagle has greater frequency. [C2]\n"
            "Eagle appears once. [C1]\n"
            "Eagle appears again. [C1]",
        )

    def test_broad_single_token_overlap_does_not_answer_specific_question(self):
        records = [self.ranked_record(0, text="Eagles ingest lead fragments.")]
        result = resolve_extractive_result("Do eagles fly at night?", records)
        self.assertTrue(result["is_fallback"])
        self.assertEqual(result["raw_answer"], INSUFFICIENT_CONTEXT_FALLBACK)

    def test_two_term_question_requires_both_terms_for_extractive_support(self):
        records = [self.ranked_record(0, text="Eagles ingest lead fragments.")]
        result = resolve_extractive_result("Eagles habitat?", records)
        self.assertTrue(result["is_fallback"])
        self.assertEqual(result["raw_answer"], INSUFFICIENT_CONTEXT_FALLBACK)

    def test_one_term_question_remains_extractive_compatible(self):
        records = [self.ranked_record(0, text="Eagles ingest lead fragments.")]
        result = resolve_extractive_result("Eagles?", records)
        self.assertFalse(result["is_fallback"])
        self.assertIn("Eagles ingest lead fragments.", result["raw_answer"])

    def test_multi_term_same_topic_without_answer_falls_back(self):
        records = [
            self.ranked_record(0, text="Bald eagles ingest lead fragments.")
        ]
        result = resolve_extractive_result(
            "How do bald eagles build winter nests?", records
        )
        self.assertTrue(result["is_fallback"])

    def test_missing_key_terms_conservatively_falls_back(self):
        records = [self.ranked_record(0, text="Bald eagles have broad wings.")]
        result = resolve_extractive_result(
            "Do bald eagles not ingest lead?", records
        )
        self.assertTrue(result["is_fallback"])

    def test_genuinely_supported_specific_question_remains_answerable(self):
        records = [
            self.ranked_record(0, text="Bald eagles ingest lead fragments.")
        ]
        result = resolve_extractive_result(
            "How do bald eagles ingest lead fragments?", records
        )
        self.assertFalse(result["is_fallback"])
        self.assertIn("Bald eagles ingest lead fragments.", result["raw_answer"])

    def test_source_sentences_with_controlled_marker_syntax_are_skipped(self):
        marker_cases = ("[C1]", "[C9]", "[C2]", "[c1]", "[C-1]", "[C 1]", "[C]")
        for marker in marker_cases:
            records = [
                self.ranked_record(
                    0,
                    text=f"Bald eagles ingest lead fragments {marker}.",
                )
            ]
            with self.subTest(marker=marker):
                result = resolve_extractive_result(
                    "How do bald eagles ingest lead fragments?", records
                )
                self.assertTrue(result["is_fallback"])
                self.assertEqual(result["citation_ids"], [])

    def test_ordinary_bracket_text_is_preserved_as_source_prose(self):
        sentence = "Bald eagles ingest lead fragments [CITES Appendix I]."
        records = [self.ranked_record(0, text=sentence)]
        result = resolve_extractive_result(
            "How do bald eagles ingest lead fragments?", records
        )
        self.assertFalse(result["is_fallback"])
        self.assertIn(sentence, result["rendered_answer"])

    def test_real_corpus_extractive_resolution_smoke(self):
        question = "How are bald eagles exposed to lead?"
        records = retrieve_locally(question, index_dir="index", top_k=4)
        result = resolve_extractive_result(question, records)
        self.assertFalse(result["is_fallback"])
        self.assertEqual(result["answer_mode"], "offline_extractive")
        self.assertNotRegex(result["rendered_answer"], r"\[C[1-9]\d*\]")
        self.assertGreater(len(result["citations"]), 0)
        self.assertEqual(
            result["unique_citation_count"], len(result["citation_ids"])
        )

    def test_deduplicates_normalized_sentence_text(self):
        records = [
            self.ranked_record(0, rank=1, text="Eagles   eat fish."),
            self.ranked_record(1, rank=2, text="  eagles eat FISH.  Another eagles fact."),
        ]
        result = build_extractive_rag_result("eagles fish fact", records)
        self.assertEqual(result["answer"].count("eat"), 1)
        self.assertIn("Another eagles fact. [C2]", result["answer"])

    def test_selects_at_most_three_sentences_and_600_source_characters(self):
        text = " ".join(f"Eagle evidence sentence {i}." for i in range(1, 8))
        result = build_extractive_rag_result(
            "eagle evidence", [self.ranked_record(0, text=text)]
        )
        self.assertEqual(result["answer"].count("[C1]"), 3)
        source_text = result["answer"].replace(" [C1]", "").replace("\n", "")
        self.assertLessEqual(len(source_text), 600)

    def test_cumulative_source_text_budget_never_exceeds_600_characters(self):
        sentences = [
            self.padded_sentence("Eagle alpha ", 250),
            self.padded_sentence("Eagle beta ", 250),
            self.padded_sentence("Eagle gamma ", 250),
        ]
        result = build_extractive_rag_result(
            "eagle", [self.ranked_record(0, text=" ".join(sentences))]
        )
        self.assertEqual(
            result["answer"],
            f"{sentences[0]} [C1]\n{sentences[1]} [C1]",
        )
        self.assertEqual(sum(len(sentence) for sentence in sentences[:2]), 500)
        self.assertNotIn(sentences[2], result["answer"])

    def test_over_remaining_budget_is_skipped_and_later_fitting_sentence_selected(self):
        selected_first = self.padded_sentence("Eagle alpha ", 400)
        skipped = self.padded_sentence("Eagle beta ", 250)
        selected_later = self.padded_sentence("Eagle gamma ", 150)
        result = build_extractive_rag_result(
            "eagle",
            [
                self.ranked_record(
                    0,
                    text=f"{selected_first} {skipped} {selected_later}",
                )
            ],
        )
        self.assertEqual(
            result["answer"],
            f"{selected_first} [C1]\n{selected_later} [C1]",
        )
        self.assertNotIn(skipped, result["answer"])
        self.assertEqual(len(selected_first) + len(selected_later), 550)

    def test_skips_over_budget_sentence_without_truncating(self):
        too_long = "Eagle " + ("x" * 594) + "."
        fitting = "Eagle fits."
        result = build_extractive_rag_result(
            "eagle", [self.ranked_record(0, text=f"{too_long} {fitting}")]
        )
        self.assertEqual(result["answer"], "Eagle fits. [C1]")
        self.assertNotIn("xxx", result["answer"])

    def test_preserves_source_sentence_verbatim_and_assigns_source_ids(self):
        records = [
            self.ranked_record(0, rank=1, text="Eagle's wings—broad & strong!"),
            self.ranked_record(1, rank=2, text="An eagle nests nearby."),
        ]
        original = copy.deepcopy(records)
        result = build_extractive_rag_result("eagle wings nests", records)
        self.assertEqual(
            result["answer"],
            "Eagle's wings—broad & strong! [C1]\nAn eagle nests nearby. [C2]",
        )
        self.assertEqual(records, original)
        self.assertEqual(result["answer_mode"], "offline_extractive")
        self.assertIsNone(result["usage_metadata"])

    def test_unsupported_builder_and_exact_final_fallback(self):
        self.assertIsNone(build_extractive_rag_result("eagle", []))
        unsupported = [self.ranked_record(0, text="Whales migrate.")]
        self.assertIsNone(build_extractive_rag_result("eagle", unsupported))

        expected = {
            "question": "eagle",
            "raw_answer": INSUFFICIENT_CONTEXT_FALLBACK,
            "rendered_answer": INSUFFICIENT_CONTEXT_FALLBACK,
            "generation_model": DEFAULT_GENERATION_MODEL,
            "citation_ids": [],
            "citations": [],
            "citation_group_count": 0,
            "unique_citation_count": 0,
            "is_fallback": True,
            "answer_mode": "offline_extractive",
        }
        self.assertEqual(build_offline_fallback_result("eagle"), expected)
        self.assertEqual(resolve_extractive_result("eagle", unsupported), expected)
        self.assertEqual(resolve_extractive_result("eagle", []), expected)

    def test_extractive_and_fallback_resolution_make_no_client_or_network_calls(self):
        supported = [
            self.ranked_record(0, text="Bald eagles ingest lead fragments.")
        ]
        with (
            patch(
                "src.generation.create_genai_client",
                side_effect=AssertionError("client creation must remain offline"),
            ) as create_client,
            patch(
                "src.retrieval.embed_query",
                side_effect=AssertionError("network embedding must remain offline"),
            ) as embed_query,
            patch(
                "src.generation.generate_grounded_answer",
                side_effect=AssertionError("network generation must remain offline"),
            ) as generate_answer,
        ):
            supported_result = resolve_extractive_result("eagles lead", supported)
            fallback_result = resolve_extractive_result("eagles lead", [])

        self.assertFalse(supported_result["is_fallback"])
        self.assertTrue(fallback_result["is_fallback"])
        create_client.assert_not_called()
        embed_query.assert_not_called()
        generate_answer.assert_not_called()

    def test_supported_result_resolves_citations_and_preserves_mode(self):
        records = [
            self.ranked_record(
                0,
                text="Bald eagles can ingest lead fragments.",
                page_number=7,
            )
        ]
        stage9 = build_extractive_rag_result("How do eagles ingest lead?", records)
        self.assertEqual(stage9["generation_model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(stage9["retrieved_context_count"], 1)
        self.assertEqual(list(stage9["source_map"]), ["C1"])

        final = resolve_extractive_result("How do eagles ingest lead?", records)
        self.assertEqual(
            final["raw_answer"], "Bald eagles can ingest lead fragments. [C1]"
        )
        self.assertEqual(
            final["rendered_answer"],
            "Bald eagles can ingest lead fragments. (document-0.pdf, p. 7)",
        )
        self.assertEqual(final["citation_ids"], ["C1"])
        self.assertEqual(final["citations"][0]["page_number"], 7)
        self.assertFalse(final["is_fallback"])
        self.assertEqual(final["answer_mode"], "offline_extractive")


class QuotaError(Exception):
    """Provider-shaped quota failure used to exercise the orchestration boundary."""

    code = 429


class TestHybridOrchestration(unittest.TestCase):
    def setUp(self):
        self.question = "How do bald eagles ingest lead fragments?"
        self.semantic_records = [
            make_record(
                0,
                text="Bald eagles ingest lead fragments from ammunition.",
            )
        ]
        self.semantic_records[0].update(rank=1, similarity_score=0.9)

    def test_normal_gemini_result_is_citation_resolved_and_marked_gemini(self):
        client = MagicMock()
        original_records = copy.deepcopy(self.semantic_records)
        with (
            patch("src.offline_fallback.retrieve", return_value=self.semantic_records) as retrieve,
            patch("src.offline_fallback.create_genai_client") as create_client,
            patch(
                "src.offline_fallback.generate_grounded_answer",
                return_value={
                    "answer": "Bald eagles ingest lead fragments from ammunition. [C1]",
                    "usage_metadata": {"total_token_count": 13},
                },
            ) as generate,
        ):
            result = answer_with_free_fallback(
                self.question,
                top_k=1,
                index_dir="semantic-index",
                api_key="test-key",
                client=client,
            )

        retrieve.assert_called_once_with(
            question=self.question,
            index_dir="semantic-index",
            top_k=1,
            api_key="test-key",
            client=client,
        )
        create_client.assert_not_called()
        generate.assert_called_once()
        self.assertEqual(result["answer_mode"], "gemini")
        self.assertNotIn("usage_metadata", result)
        self.assertEqual(result["citation_ids"], ["C1"])
        self.assertEqual(self.semantic_records, original_records)
        self.assertEqual(
            result["rendered_answer"],
            "Bald eagles ingest lead fragments from ammunition. (document-0.pdf, p. 1)",
        )

    def test_generation_quota_reuses_semantic_records_without_local_retrieval(self):
        client = MagicMock()
        offline_result = {"answer_mode": "offline_extractive", "is_fallback": False}
        with (
            patch("src.offline_fallback.retrieve", return_value=self.semantic_records),
            patch("src.offline_fallback.create_genai_client", return_value=client),
            patch(
                "src.offline_fallback.generate_grounded_answer",
                side_effect=QuotaError("rate limit"),
            ) as generate,
            patch("src.offline_fallback.retrieve_locally") as retrieve_locally,
            patch(
                "src.offline_fallback.resolve_extractive_result",
                return_value=offline_result,
            ) as resolve_extractive,
        ):
            result = answer_with_free_fallback(self.question, top_k=1)

        self.assertIs(result, offline_result)
        generate.assert_called_once()
        retrieve_locally.assert_not_called()
        resolve_extractive.assert_called_once_with(self.question, self.semantic_records)

    def test_retrieval_quota_uses_local_retrieval_without_generation(self):
        local_records = [
            dict(self.semantic_records[0], rank=1, similarity_score=1.0)
        ]
        offline_result = {"answer_mode": "offline_extractive", "is_fallback": False}
        with (
            patch(
                "src.offline_fallback.retrieve",
                side_effect=QuotaError("resource exhausted"),
            ),
            patch(
                "src.offline_fallback.retrieve_locally", return_value=local_records
            ) as retrieve_locally,
            patch("src.offline_fallback.create_genai_client") as create_client,
            patch("src.offline_fallback.generate_grounded_answer") as generate,
            patch(
                "src.offline_fallback.resolve_extractive_result",
                return_value=offline_result,
            ) as resolve_extractive,
        ):
            result = answer_with_free_fallback(self.question, top_k=1, index_dir="local-index")

        self.assertIs(result, offline_result)
        retrieve_locally.assert_called_once_with(self.question, "local-index", 1)
        resolve_extractive.assert_called_once_with(self.question, local_records)
        create_client.assert_not_called()
        generate.assert_not_called()

    def test_retrieval_quota_no_local_support_returns_exact_offline_fallback_without_network(self):
        expected = build_offline_fallback_result(self.question)
        with (
            patch("src.offline_fallback.retrieve", side_effect=QuotaError("quota")),
            patch("src.offline_fallback.retrieve_locally", return_value=[]) as retrieve_locally,
            patch("src.offline_fallback.create_genai_client") as create_client,
            patch("src.offline_fallback.generate_grounded_answer") as generate,
        ):
            result = answer_with_free_fallback(self.question, top_k=1, index_dir="local-index")

        self.assertEqual(result, expected)
        self.assertEqual(result["answer_mode"], "offline_extractive")
        retrieve_locally.assert_called_once_with(self.question, "local-index", 1)
        create_client.assert_not_called()
        generate.assert_not_called()

    def test_non_quota_retrieval_error_propagates_without_local_fallback(self):
        error = RuntimeError("authentication failed")
        with (
            patch("src.offline_fallback.retrieve", side_effect=error),
            patch("src.offline_fallback.retrieve_locally") as retrieve_locally,
        ):
            with self.assertRaises(RuntimeError) as caught:
                answer_with_free_fallback(self.question, top_k=1)

        self.assertIs(caught.exception, error)
        retrieve_locally.assert_not_called()

    def test_non_quota_generation_error_propagates_without_local_fallback(self):
        error = RuntimeError("authentication failed")
        with (
            patch("src.offline_fallback.retrieve", return_value=self.semantic_records),
            patch("src.offline_fallback.create_genai_client", return_value=MagicMock()),
            patch("src.offline_fallback.generate_grounded_answer", side_effect=error),
            patch("src.offline_fallback.retrieve_locally") as retrieve_locally,
        ):
            with self.assertRaises(RuntimeError) as caught:
                answer_with_free_fallback(self.question, top_k=1)

        self.assertIs(caught.exception, error)
        retrieve_locally.assert_not_called()


if __name__ == "__main__":
    unittest.main()
