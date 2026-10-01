"""Unit tests for Stage 5 chunking module."""

import unittest
from src.chunking import (
    HARD_MAX_TOKENS,
    OVERLAP_TOKENS,
    SMALL_CHUNK_THRESHOLD,
    TARGET_TOKENS,
    _compute_overlap_text,
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
        # Deterministically generate 2 full chunks by exceeding TARGET_TOKENS (450 tok)
        p1_sentences = [f"DistinctSentenceA{i:02d} provides initial long body content." for i in range(25)]
        s_tail = "ThisIsBoundaryOverlapPassage sentence that bridges chunk one and chunk two."
        p1 = " ".join(p1_sentences) + " " + s_tail  # ~440 tokens

        p2_sentences = [f"DistinctSentenceB{i:02d} provides second chunk body content." for i in range(25)]
        p2 = " ".join(p2_sentences)  # ~440 tokens

        page = {
            "document_id": "doc_overlap_test",
            "filename": "test.pdf",
            "page_number": 1,
            "text": f"{p1}\n\n{p2}",
            "title": "Title",
            "publisher": "Pub",
            "source_url": "http://example.com",
        }

        chunks = chunk_processed_page(page)

        # Unconditional assertions
        self.assertGreaterEqual(len(chunks), 2, "Fixture must produce at least two chunks")

        c1_text = chunks[0]["text"]
        c2_text = chunks[1]["text"]

        self.assertNotEqual(c1_text, c2_text, "Adjacent chunks must not be identical")

        # Calculate full expected overlap using _compute_overlap_text
        expected_overlap = _compute_overlap_text(c1_text, OVERLAP_TOKENS)

        self.assertTrue(bool(expected_overlap), "Expected overlap must be nonempty")
        self.assertTrue(c2_text.startswith(expected_overlap), "Chunk 2 must start with full expected overlap")
        self.assertLessEqual(estimate_tokens(expected_overlap), OVERLAP_TOKENS, "Overlap token budget must be respected")

        # Verify second chunk advances source content
        self.assertIn("DistinctSentenceB00", c2_text, "Chunk 2 must contain new source content")

        # Verify hard max compliance
        for c in chunks:
            self.assertLessEqual(c["estimated_token_count"], HARD_MAX_TOKENS)

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

    def test_10_small_tail_merge_prevents_duplicate_overlap(self):
        # Construct deterministic fixture using 27 sentences in block 1:
        # Block 1 (~440 tok) + Block 2 (~16 tok).
        # Block 1 alone <= TARGET_TOKENS (450 tok).
        # Block 1 + Block 2 > TARGET_TOKENS (450 tok), forcing a chunk split.
        # Overlapped tail is < SMALL_CHUNK_THRESHOLD (100 tok), triggering small-tail merge.
        b1_sentences = [f"Sentence{i:02d} in block one contains unique detailed body content." for i in range(27)]
        s_tail = "UniqueSentenceTailPassage bridging the small tail merge test."
        b1 = " ".join(b1_sentences) + " " + s_tail
        b2 = "SentenceZeroOne in block two represents tiny new ending content."

        b1_plus_b2 = f"{b1}\n\n{b2}"
        overlap_str = _compute_overlap_text(b1, OVERLAP_TOKENS)
        overlapped_tail = f"{overlap_str}\n\n{b2}"

        # Explicit precondition assertions
        self.assertLessEqual(estimate_tokens(b1), TARGET_TOKENS, "Precondition: b1 alone must be <= TARGET_TOKENS")
        self.assertGreater(estimate_tokens(b1_plus_b2), TARGET_TOKENS, "Precondition: b1 + b2 must exceed TARGET_TOKENS")
        self.assertTrue(bool(overlap_str), "Precondition: computed overlap must be nonempty")
        self.assertLess(estimate_tokens(overlapped_tail), SMALL_CHUNK_THRESHOLD, "Precondition: overlapped tail must be < SMALL_CHUNK_THRESHOLD")
        self.assertLessEqual(estimate_tokens(b1_plus_b2), HARD_MAX_TOKENS, "Precondition: merged source must be <= HARD_MAX_TOKENS")

        page = {
            "document_id": "doc_regression_test",
            "filename": "test.pdf",
            "page_number": 1,
            "text": b1_plus_b2,
            "title": "Title",
            "publisher": "Pub",
            "source_url": "http://example.com",
        }

        chunks = chunk_processed_page(page)

        # Unconditional assertions
        self.assertEqual(len(chunks), 1, "Exactly one final chunk should remain after small-tail merge")

        merged_text = chunks[0]["text"]

        self.assertEqual(merged_text.count(s_tail), 1, "Boundary overlap passage must appear exactly once in merged chunk")
        self.assertEqual(merged_text.count("SentenceZeroOne"), 1, "Tiny ending content must appear exactly once")

        # Verify content ordering
        pos_b1 = merged_text.find("Sentence00")
        pos_tail = merged_text.find(s_tail)
        pos_b2 = merged_text.find("SentenceZeroOne")
        self.assertTrue(0 <= pos_b1 < pos_tail < pos_b2, "All intended source content must remain in correct order")

        # Verify hard maximum compliance
        self.assertLessEqual(chunks[0]["estimated_token_count"], HARD_MAX_TOKENS, "Merged chunk must respect HARD_MAX_TOKENS")

        # Compare complete output against expected source text with documented separators
        self.assertEqual(merged_text, b1_plus_b2, "Merged text must match source paragraphs joined by double newline")

    def test_11_natural_source_repetition_preserved(self):
        # Verify naturally repeated source text (e.g. echo echo) is NOT stripped
        para = "The wolf called out echo echo echo across the valley."
        page = dict(self.sample_page, text=para)
        chunks = chunk_processed_page(page)

        self.assertEqual(len(chunks), 1)
        self.assertIn("echo echo echo", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
