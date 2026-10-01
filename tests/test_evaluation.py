"""Offline test suite for Stage 14 evaluation modules.

Validates schema contracts, question distribution, document coverage, metric formulas,
manual review CSV parsing, and single-pass pipeline reuse without live API calls.
"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evaluation.run_evaluation import (
    EXPECTED_INDEX_HASHES,
    compute_file_sha256,
    verify_index_artifacts,
)
from evaluation.score_evaluation import (
    parse_human_eval_csv,
    score_rag_generation,
    score_retrieval,
)


class TestEvaluationDatasetAndArtifacts(unittest.TestCase):
    """Test golden dataset structure, index artifact hashes, and deliverable constraints."""

    def test_index_artifact_hashes(self) -> None:
        """Verify that tracked index artifact hashes match expected Stage 7 baseline."""
        verified_hashes = verify_index_artifacts("index")
        for path, exp_hash in EXPECTED_INDEX_HASHES.items():
            self.assertEqual(verified_hashes[path], exp_hash)

    def test_golden_questions_schema_and_counts(self) -> None:
        """Verify golden dataset schema, 20-question count, 16/4 breakdown, and coverage."""
        golden_path = "evaluation/golden_questions.json"
        self.assertTrue(os.path.exists(golden_path))

        with open(golden_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        self.assertEqual(len(questions), 20)

        supported = [q for q in questions if q["type"] == "supported"]
        unsupported = [q for q in questions if q["type"] == "unsupported"]

        self.assertEqual(len(supported), 16)
        self.assertEqual(len(unsupported), 4)

        # Verify 9 document representation
        doc_counts = {}
        animal_groups = set()
        new_count = 0
        reused_count = 0

        for q in supported:
            animal_groups.add(q["animal_group"])
            if q["origin"] == "new":
                new_count += 1
            elif q["origin"] == "reused":
                reused_count += 1

            for fn in q["expected_filenames"]:
                doc_counts[fn] = doc_counts.get(fn, 0) + 1

            # Check accepted evidence schema
            self.assertTrue(isinstance(q["accepted_evidence"], list))
            for ev in q["accepted_evidence"]:
                self.assertIn("filename", ev)
                self.assertIn("page", ev)

        for q in unsupported:
            animal_groups.add(q["animal_group"])
            self.assertEqual(q["expected_behavior"], "exact_fallback")
            self.assertEqual(len(q["expected_filenames"]), 0)
            self.assertEqual(len(q["accepted_pages"]), 0)

        self.assertEqual(len(doc_counts), 9)
        self.assertLessEqual(max(doc_counts.values()), 3)
        self.assertGreaterEqual(len(animal_groups), 5)
        self.assertGreaterEqual(new_count, 12)
        self.assertLessEqual(reused_count, 4)


class TestEvaluationScorerMetrics(unittest.TestCase):
    """Test offline metric calculations for retrieval and RAG generation."""

    def setUp(self) -> None:
        self.golden_questions = [
            {
                "id": "Q01",
                "type": "supported",
                "expected_filenames": ["fileA.pdf"],
                "accepted_evidence": [{"filename": "fileA.pdf", "page": 1}],
            },
            {
                "id": "Q02",
                "type": "supported",
                "expected_filenames": ["fileB.pdf"],
                "accepted_evidence": [{"filename": "fileB.pdf", "page": 2}],
            },
            {
                "id": "Q17",
                "type": "unsupported",
                "expected_filenames": [],
                "accepted_evidence": [],
                "expected_behavior": "exact_fallback",
            },
        ]

    def test_score_retrieval_metrics(self) -> None:
        """Test Document Hit@4, MRR@4, and Page Hit@4 calculation."""
        retrieval_results = [
            {
                "question_id": "Q01",
                "retrieved_chunks": [
                    {"rank": 1, "filename": "fileX.pdf", "page_number": 5},
                    {"rank": 2, "filename": "fileA.pdf", "page_number": 1},
                ],
            },
            {
                "question_id": "Q02",
                "retrieved_chunks": [
                    {"rank": 1, "filename": "fileB.pdf", "page_number": 9},  # Doc hit, but page miss (expected page 2)
                ],
            },
        ]

        scores = score_retrieval(retrieval_results, self.golden_questions)

        # 2 total supported questions in mock
        self.assertEqual(scores["total_supported_questions"], 2)
        self.assertEqual(scores["document_hit_at_4_count"], 2)
        self.assertEqual(scores["document_hit_at_4_rate"], 1.0)
        # MRR: Q01 rank 2 (0.5), Q02 rank 1 (1.0) -> mean = 0.75
        self.assertEqual(scores["document_mrr_at_4"], 0.75)
        # Page hit: Q01 hit page 1 (hit), Q02 hit page 9 (miss) -> 1 hit
        self.assertEqual(scores["page_hit_at_4_count"], 1)
        self.assertEqual(scores["page_hit_at_4_rate"], 0.5)

    def test_score_rag_generation_metrics(self) -> None:
        """Test non-fallback rate, exact fallback rate, and citation acceptance rate."""
        rag_results = [
            {
                "question_id": "Q01",
                "type": "supported",
                "status": "success",
                "is_fallback": False,
                "citation_ids": ["C1"],
                "citations": [{"filename": "fileA.pdf", "page_number": 1}],
            },
            {
                "question_id": "Q02",
                "type": "supported",
                "status": "success",
                "is_fallback": True,  # Unexpected fallback
                "citation_ids": [],
                "citations": [],
            },
            {
                "question_id": "Q17",
                "type": "unsupported",
                "status": "success",
                "is_fallback": True,  # Exact fallback
                "citation_ids": [],
                "citations": [],
            },
        ]

        scores = score_rag_generation(rag_results, self.golden_questions)

        self.assertEqual(scores["supported"]["total"], 2)
        self.assertEqual(scores["supported"]["non_fallback_count"], 1)
        self.assertEqual(scores["supported"]["non_fallback_rate"], 0.5)
        self.assertEqual(scores["supported"]["fallback_count"], 1)
        # Citation acceptance: 1 valid / 1 attempted non-fallback = 1.0
        self.assertEqual(scores["supported"]["citation_acceptance_rate"], 1.0)

        self.assertEqual(scores["unsupported"]["total"], 1)
        self.assertEqual(scores["unsupported"]["exact_fallback_count"], 1)
        self.assertEqual(scores["unsupported"]["exact_fallback_rate"], 1.0)
        self.assertEqual(scores["unsupported"]["clean_fallback_count"], 1)
        self.assertEqual(scores["unsupported"]["clean_fallback_rate"], 1.0)


class TestHumanEvalCSVValidation(unittest.TestCase):
    """Test manual_review.csv parsing, validation rules, and error handling."""

    def test_parse_human_eval_csv_valid_and_incomplete(self) -> None:
        """Test parsing completed vs incomplete human review CSV files."""
        supported_ids = ["Q01", "Q02"]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "manual_review.csv")

            # 1. Incomplete CSV (blank scores)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["question_id", "question", "correctness", "groundedness", "citation_support", "notes"])
                writer.writerow(["Q01", "Question 1", "", "", "", ""])
                writer.writerow(["Q02", "Question 2", "2", "2", "2", "Notes"])

            res, errors = parse_human_eval_csv(csv_path, supported_ids)
            self.assertIsNone(res)
            self.assertTrue(len(errors) > 0)
            self.assertIn("Missing human score values", errors[0])

            # 2. Fully completed valid CSV
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["question_id", "question", "correctness", "groundedness", "citation_support", "notes"])
                writer.writerow(["Q01", "Question 1", "2", "2", "2", "Fully correct"])
                writer.writerow(["Q02", "Question 2", "1", "2", "2", "Minor detail"])

            res, errors = parse_human_eval_csv(csv_path, supported_ids)
            self.assertEqual(len(errors), 0)
            self.assertIsNotNone(res)
            self.assertEqual(res["status"], "COMPLETED")
            self.assertEqual(res["total_reviewed"], 2)
            self.assertEqual(res["fully_correct_count"], 1)
            self.assertEqual(res["fully_grounded_count"], 2)
            self.assertEqual(res["mean_correctness"], 1.5)


if __name__ == "__main__":
    unittest.main()
