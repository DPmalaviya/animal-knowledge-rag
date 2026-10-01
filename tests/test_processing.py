"""Unit tests for Stage 5 text processing module."""

import unittest
from src.processing import normalize_text, process_page, process_pages


class TestProcessing(unittest.TestCase):
    """Test suite covering text normalization and page record processing."""

    def test_01_normalization_returns_string(self):
        result = normalize_text("Hello world.")
        self.assertIsInstance(result, str)

    def test_02_repeated_whitespace_handled(self):
        raw = "Line  with   multiple   spaces.  \t  And tabs."
        expected = "Line with multiple spaces. And tabs."
        self.assertEqual(normalize_text(raw), expected)

    def test_03_meaningful_paragraph_boundaries_survive(self):
        raw = "Paragraph 1.\n\n\n\nParagraph 2."
        expected = "Paragraph 1.\n\nParagraph 2."
        self.assertEqual(normalize_text(raw), expected)

    def test_04_legitimate_hyphenated_terms_preserved(self):
        raw = "scent-detection dog and non-lead ammunition."
        normalized = normalize_text(raw)
        self.assertIn("scent-detection", normalized)
        self.assertIn("non-lead", normalized)

    def test_05_processing_preserves_metadata(self):
        record = {
            "document_id": "doc_test",
            "filename": "test.pdf",
            "page_number": 1,
            "text": "Some test text content.",
            "title": "Test Title",
            "publisher": "Test Publisher",
            "source_url": "http://example.com/test",
        }
        processed = process_page(record)
        self.assertEqual(processed["document_id"], "doc_test")
        self.assertEqual(processed["filename"], "test.pdf")
        self.assertEqual(processed["page_number"], 1)
        self.assertEqual(processed["title"], "Test Title")
        self.assertEqual(processed["publisher"], "Test Publisher")
        self.assertEqual(processed["source_url"], "http://example.com/test")

    def test_06_input_page_records_not_mutated(self):
        raw_text = "   Unnormalized   text.   "
        record = {
            "document_id": "doc_test",
            "filename": "test.pdf",
            "page_number": 1,
            "text": raw_text,
            "title": "Title",
            "publisher": "Pub",
            "source_url": "http://example.com",
        }
        processed = process_page(record)
        self.assertEqual(record["text"], raw_text)
        self.assertNotEqual(processed["text"], raw_text)

    def test_07_empty_whitespace_input_explicit_behavior(self):
        with self.assertRaises(ValueError):
            normalize_text("   \n\t  ")

    def test_08_duplicate_source_pages_rejected(self):
        records = [
            {
                "document_id": "doc_test",
                "filename": "test.pdf",
                "page_number": 1,
                "text": "Page 1 text.",
                "title": "Title",
                "publisher": "Pub",
                "source_url": "http://example.com",
            },
            {
                "document_id": "doc_test",
                "filename": "test.pdf",
                "page_number": 1,
                "text": "Duplicate Page 1 text.",
                "title": "Title",
                "publisher": "Pub",
                "source_url": "http://example.com",
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            process_pages(records)
        self.assertIn("Duplicate source page identity detected", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
