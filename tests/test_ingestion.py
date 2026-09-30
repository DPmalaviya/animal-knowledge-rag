"""Unit tests for Stage 4 document ingestion module."""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf

from src.ingestion import (
    REQUIRED_MANIFEST_COLUMNS,
    extract_pdf_pages,
    ingest_corpus,
    load_manifest,
    validate_dataset,
)


class TestIngestion(unittest.TestCase):
    """Test suite covering manifest validation, PDF text extraction, and corpus ingestion."""

    # 1. The real manifest loads.
    def test_01_real_manifest_loads(self):
        manifest_data = load_manifest("data/dataset_manifest.csv")
        self.assertIsInstance(manifest_data, list)
        self.assertEqual(len(manifest_data), 9)

    # 2. Required metadata survives extraction unchanged.
    def test_02_metadata_survives_extraction_unchanged(self):
        records = ingest_corpus("data/dataset_manifest.csv", "data/raw")
        manifest_data = load_manifest("data/dataset_manifest.csv")
        manifest_map = {row["document_id"]: row for row in manifest_data}

        for record in records:
            doc_id = record["document_id"]
            self.assertIn(doc_id, manifest_map)
            m_row = manifest_map[doc_id]
            self.assertEqual(record["filename"], m_row["filename"])
            self.assertEqual(record["title"], m_row["title"])
            self.assertEqual(record["publisher"], m_row["publisher"])
            self.assertEqual(record["source_url"], m_row["source_url"])

    # 3. All manifest PDFs exist.
    def test_03_all_manifest_pdfs_exist(self):
        manifest_data = load_manifest("data/dataset_manifest.csv")
        raw_dir = Path("data/raw")
        for row in manifest_data:
            pdf_path = raw_dir / row["filename"]
            self.assertTrue(pdf_path.is_file(), f"Missing PDF: {pdf_path}")

    # 4. Unlisted PDFs are detected.
    def test_04_unlisted_pdfs_detected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_tmp = tmp_path / "raw"
            raw_tmp.mkdir()

            # Copy a real PDF
            real_pdf = Path("data/raw/monarch_butterfly_factsheet.pdf")
            shutil.copy(real_pdf, raw_tmp / real_pdf.name)

            # Create unlisted PDF
            unlisted_pdf = raw_tmp / "unlisted_extra_doc.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Unlisted test content")
            doc.save(str(unlisted_pdf))
            doc.close()

            manifest_data = [{
                "document_id": "doc_monarch_butterfly",
                "filename": "monarch_butterfly_factsheet.pdf",
                "title": "Title",
                "publisher": "Pub",
                "source_url": "http://example.com",
                "page_count": "1",
            }]

            with self.assertRaises(ValueError) as ctx:
                validate_dataset(manifest_data, str(raw_tmp))
            self.assertIn("Unmanifested PDF file(s) found", str(ctx.exception))
            self.assertIn("unlisted_extra_doc.pdf", str(ctx.exception))

    # 5. Duplicate document IDs are rejected.
    def test_05_duplicate_doc_ids_rejected(self):
        manifest_data = [
            {
                "document_id": "doc_dup",
                "filename": "bald_eagle_factsheet.pdf",
                "title": "T1",
                "publisher": "P1",
                "source_url": "S1",
                "page_count": "2",
            },
            {
                "document_id": "doc_dup",
                "filename": "monarch_butterfly_factsheet.pdf",
                "title": "T2",
                "publisher": "P2",
                "source_url": "S2",
                "page_count": "1",
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(manifest_data, "data/raw")
        self.assertIn("Duplicate document_id found in manifest: 'doc_dup'", str(ctx.exception))

    # 6. Duplicate filenames are rejected.
    def test_06_duplicate_filenames_rejected(self):
        manifest_data = [
            {
                "document_id": "doc1",
                "filename": "bald_eagle_factsheet.pdf",
                "title": "T1",
                "publisher": "P1",
                "source_url": "S1",
                "page_count": "2",
            },
            {
                "document_id": "doc2",
                "filename": "bald_eagle_factsheet.pdf",
                "title": "T2",
                "publisher": "P2",
                "source_url": "S2",
                "page_count": "2",
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_dataset(manifest_data, "data/raw")
        self.assertIn("Duplicate filename found in manifest: 'bald_eagle_factsheet.pdf'", str(ctx.exception))

    # 7. A valid PDF opens and produces text.
    def test_07_valid_pdf_opens_and_produces_text(self):
        meta = {
            "document_id": "doc_monarch_butterfly",
            "filename": "monarch_butterfly_factsheet.pdf",
            "title": "Fly into Action",
            "publisher": "USFWS",
            "source_url": "https://www.fws.gov",
        }
        records = extract_pdf_pages("data/raw/monarch_butterfly_factsheet.pdf", meta)
        self.assertEqual(len(records), 1)
        self.assertIn("Monarch", records[0]["text"])

    # 8. Page numbering begins at 1 and remains contiguous per document.
    def test_08_page_numbering_contiguous_from_1(self):
        records = ingest_corpus("data/dataset_manifest.csv", "data/raw")
        doc_pages = {}
        for r in records:
            doc_id = r["document_id"]
            doc_pages.setdefault(doc_id, []).append(r["page_number"])

        for doc_id, pages in doc_pages.items():
            self.assertEqual(pages, list(range(1, len(pages) + 1)), f"Non-contiguous pages in {doc_id}")

    # 9. Extracted text is nonempty for every approved corpus page.
    def test_09_extracted_text_nonempty(self):
        records = ingest_corpus("data/dataset_manifest.csv", "data/raw")
        for r in records:
            self.assertTrue(bool(r["text"].strip()), f"Empty text on page {r['page_number']} of {r['filename']}")

    # 10. Actual per-document page counts match the manifest.
    def test_10_actual_page_counts_match_manifest(self):
        manifest_data = load_manifest("data/dataset_manifest.csv")
        records = ingest_corpus("data/dataset_manifest.csv", "data/raw")

        doc_record_counts = {}
        for r in records:
            doc_record_counts[r["document_id"]] = doc_record_counts.get(r["document_id"], 0) + 1

        for row in manifest_data:
            doc_id = row["document_id"]
            expected = int(row["page_count"])
            actual = doc_record_counts[doc_id]
            self.assertEqual(actual, expected, f"Page count mismatch for {doc_id}")

    # 11. Full-corpus ingestion yields exactly 9 distinct documents and 85 page records.
    def test_11_full_corpus_yields_9_docs_85_pages(self):
        records = ingest_corpus("data/dataset_manifest.csv", "data/raw")
        unique_docs = {r["document_id"] for r in records}
        self.assertEqual(len(unique_docs), 9)
        self.assertEqual(len(records), 85)

    # 12. A controlled failure produces a useful error rather than silently skipping work.
    def test_12_controlled_failure_raises_useful_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_tmp = tmp_path / "raw"
            raw_tmp.mkdir()

            # Create an empty text page PDF
            empty_pdf = raw_tmp / "empty_page.pdf"
            doc = pymupdf.open()
            doc.new_page()  # Blank page with no text
            doc.save(str(empty_pdf))
            doc.close()

            meta = {
                "document_id": "doc_empty",
                "filename": "empty_page.pdf",
                "title": "Empty Doc",
                "publisher": "Test",
                "source_url": "http://example.com",
            }

            with self.assertRaises(ValueError) as ctx:
                extract_pdf_pages(str(empty_pdf), meta)

            err_msg = str(ctx.exception)
            self.assertIn("Empty or whitespace-only page text detected", err_msg)
            self.assertIn("empty_page.pdf", err_msg)
            self.assertIn("physical page 1", err_msg)


if __name__ == "__main__":
    unittest.main()
