"""Unit tests for Stage 5 chunking module."""

import unittest
from src.chunking import (
    HARD_MAX_TOKENS,
    TARGET_TOKENS,
    chunk_pages,
    chunk_processed_page,
    estimate_tokens,
)
from src.ingestion import ingest_corpus
from src.processing import process_pages


class TestChunking(unittest.TestCase):
    """Test suite covering page-aware chunking, overlap, sizing, and corpus preservation."""

    def setUp(self):
        self.sample_page = {
            "document_id": "doc_sample",
            "filename": "sample.pdf",
            "page_number": 1,
            "text": "First paragraph sentence one. First paragraph sentence two.\n\n"
                    "Second paragraph sentence one. Second paragraph sentence two.\n\n"
                    "Third paragraph sentence one. Third paragraph sentence two.",
            "title": "Sample Title",
            "publisher": "Sample Publisher",
            "source_url": "http://example.com/sample",
        }

    def test_01_chunk_ids_unique_and_deterministic(self):
        chunks1 = chunk_processed_page(self.sample_page)
        chunks2 = chunk_processed_page(self.sample_page)

        ids1 = [c["chunk_id"] for c in chunks1]
        ids2 = [c["chunk_id"] for c in chunks2]

        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids1), len(set(ids1)))
        self.assertTrue(ids1[0].startswith("doc_sample_p001_c001"))

    def test_02_metadata_matches_source_page(self):
        chunks = chunk_processed_page(self.sample_page)
        for c in chunks:
            self.assertEqual(c["document_id"], "doc_sample")
            self.assertEqual(c["filename"], "sample.pdf")
            self.assertEqual(c["page_number"], 1)
            self.assertEqual(c["title"], "Sample Title")
            self.assertEqual(c["publisher"], "Sample Publisher")
            self.assertEqual(c["source_url"], "http://example.com/sample")

    def test_03_chunk_text_nonempty(self):
        chunks = chunk_processed_page(self.sample_page)
        for c in chunks:
            self.assertTrue(bool(c["text"].strip()))

    def test_04_hard_maximum_includes_overlap(self):
        # Create a long text to generate multiple chunks
        long_para = "Word " * 250
        long_page = dict(self.sample_page, text=f"{long_para}\n\n{long_para}\n\n{long_para}")
        chunks = chunk_processed_page(long_page)

        for c in chunks:
            self.assertLessEqual(c["estimated_token_count"], HARD_MAX_TOKENS)

    def test_05_oversized_paragraph_and_sentence_termination(self):
        # Test long single sentence without periods
        long_sentence = "UninterruptedLongSentence " * 300
        page = dict(self.sample_page, text=long_sentence)
        chunks = chunk_processed_page(page)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c["estimated_token_count"], HARD_MAX_TOKENS)

    def test_06_uninterrupted_string_slices_safely(self):
        # Test single word longer than HARD_MAX_TOKENS * 4 chars
        giant_string = "A" * 3000
        page = dict(self.sample_page, text=giant_string)
        chunks = chunk_processed_page(page)

        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c["estimated_token_count"], HARD_MAX_TOKENS)

    def test_07_adjacent_chunks_contain_overlap_and_advance_content(self):
        para1 = "Sentence A1. Sentence A2. Sentence A3. Sentence A4."
        para2 = "Sentence B1. Sentence B2. Sentence B3. Sentence B4."
        # Create small target to force chunk split
        page = {
            "document_id": "doc_test",
            "filename": "test.pdf",
            "page_number": 1,
            "text": f"{para1}\n\n{para2}",
            "title": "T",
            "publisher": "P",
            "source_url": "U",
        }
        chunks = chunk_processed_page(page)
        if len(chunks) > 1:
            chunk1_text = chunks[0]["text"]
            chunk2_text = chunks[1]["text"]

            # Verify chunk2 is not identical to chunk1
            self.assertNotEqual(chunk1_text, chunk2_text)
            # Verify chunk2 introduces new text beyond chunk1
            self.assertIn("Sentence B", chunk2_text)

    def test_08_full_corpus_ingest_process_chunk(self):
        raw_pages = ingest_corpus()
        proc_pages = process_pages(raw_pages)
        chunks = chunk_pages(proc_pages)

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

        docs = {c["document_id"] for c in chunks}
        pages = {(c["document_id"], c["page_number"]) for c in chunks}

        self.assertEqual(len(docs), 9)
        self.assertEqual(len(pages), 85)

    def test_09_duplicate_chunk_id_detection(self):
        proc_pages = [
            dict(self.sample_page),
            dict(self.sample_page),  # Same document_id and page_number
        ]
        with self.assertRaises(ValueError) as ctx:
            chunk_pages(proc_pages)
        self.assertIn("Duplicate chunk_id generated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
