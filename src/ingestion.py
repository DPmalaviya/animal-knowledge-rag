"""Document ingestion module for Animal Knowledge RAG Assistant.

Extracts page-level raw text from PDF documents using PyMuPDF and validates
dataset integrity against the approved CSV manifest.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

import pymupdf


REQUIRED_MANIFEST_COLUMNS = [
    "document_id",
    "filename",
    "title",
    "publisher",
    "source_url",
    "page_count",
]


def load_manifest(manifest_path: str = "data/dataset_manifest.csv") -> List[Dict[str, str]]:
    """Load and parse the dataset manifest CSV file into a list of row dictionaries.

    Args:
        manifest_path: Path to the CSV manifest file.

    Returns:
        List of dictionaries containing manifest row entries.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If required columns are missing from CSV header.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing_cols = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in fieldnames]
        if missing_cols:
            raise ValueError(
                f"Manifest header missing required column(s): {', '.join(missing_cols)}"
            )

        rows = list(reader)

    return rows


def validate_dataset(
    manifest_data: List[Dict[str, str]],
    raw_dir: str = "data/raw",
) -> None:
    """Validate manifest entries and raw PDF files for consistency and completeness.

    Args:
        manifest_data: Loaded manifest dictionary records.
        raw_dir: Directory containing raw PDF files.

    Raises:
        ValueError: If any validation rule fails.
        FileNotFoundError: If a listed PDF file is missing.
    """
    raw_path = Path(raw_dir).resolve()
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    seen_doc_ids = set()
    seen_filenames = set()
    manifest_filenames = set()

    for idx, row in enumerate(manifest_data, start=2):  # Row 1 is header
        # Check required fields non-blank
        for col in REQUIRED_MANIFEST_COLUMNS:
            val = (row.get(col) or "").strip()
            if not val:
                raise ValueError(
                    f"Manifest row {idx} has blank value for required column '{col}'"
                )

        doc_id = row["document_id"].strip()
        filename = row["filename"].strip()

        # Check unique IDs
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate document_id found in manifest: '{doc_id}' at row {idx}")
        seen_doc_ids.add(doc_id)

        # Check unique filenames
        if filename in seen_filenames:
            raise ValueError(f"Duplicate filename found in manifest: '{filename}' at row {idx}")
        seen_filenames.add(filename)

        manifest_filenames.add(filename)

        # File existence & scope check
        pdf_path = (Path(raw_dir) / filename).resolve()
        try:
            pdf_path.relative_to(raw_path)
        except ValueError:
            raise ValueError(
                f"Filename '{filename}' resolves outside raw directory '{raw_path}'"
            )

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF file referenced in manifest row {idx} does not exist: {pdf_path}"
            )

        # Page count check
        try:
            expected_pages = int(row["page_count"])
        except ValueError:
            raise ValueError(
                f"Invalid page_count '{row['page_count']}' in manifest row {idx} for '{filename}'"
            )

        try:
            with pymupdf.open(pdf_path) as doc:
                actual_pages = doc.page_count
        except Exception as e:
            raise ValueError(f"Failed to open/read PDF '{filename}': {e}") from e

        if actual_pages != expected_pages:
            raise ValueError(
                f"Page count mismatch for '{filename}': manifest records {expected_pages} pages, "
                f"actual PDF contains {actual_pages} pages."
            )

    # Check for unmanifested PDFs in raw directory (ignore .gitkeep and other non-PDF files)
    disk_pdfs = {p.name for p in raw_path.glob("*.pdf")}
    unmanifested = disk_pdfs - manifest_filenames
    if unmanifested:
        raise ValueError(
            f"Unmanifested PDF file(s) found in '{raw_dir}': {', '.join(sorted(unmanifested))}"
        )


def extract_pdf_pages(
    file_path: str,
    doc_metadata: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Extract raw page text and metadata for every physical page in a PDF document.

    Args:
        file_path: Path to the PDF file.
        doc_metadata: Manifest metadata dictionary for this document.

    Returns:
        List of page-record dictionaries containing document metadata and raw extracted text.

    Raises:
        FileNotFoundError: If PDF file does not exist.
        ValueError: If PDF cannot be read, has zero pages, or contains an empty page.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    records = []
    filename = path.name

    try:
        with pymupdf.open(path) as doc:
            if doc.is_encrypted:
                raise ValueError(f"Encrypted/password-protected PDF cannot be processed: '{filename}'")

            if doc.page_count == 0:
                raise ValueError(f"Zero-page document detected: '{filename}'")

            for page_idx in range(doc.page_count):
                page_num = page_idx + 1  # 1-based physical page number
                page = doc[page_idx]

                # sort=True orders text blocks top-to-bottom, left-to-right for multi-column pages
                raw_text = page.get_text("text", sort=True)

                if not raw_text.strip():
                    raise ValueError(
                        f"Empty or whitespace-only page text detected in document '{filename}' "
                        f"at physical page {page_num}."
                    )

                record = {
                    "document_id": doc_metadata["document_id"],
                    "filename": filename,
                    "page_number": page_num,
                    "text": raw_text,
                    "title": doc_metadata["title"],
                    "publisher": doc_metadata["publisher"],
                    "source_url": doc_metadata["source_url"],
                }
                records.append(record)
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        raise ValueError(f"Extraction failed for PDF '{filename}': {e}") from e

    return records


def ingest_corpus(
    manifest_path: str = "data/dataset_manifest.csv",
    raw_dir: str = "data/raw",
) -> List[Dict[str, Any]]:
    """Ingest the full dataset corpus by loading manifest, validating data, and extracting pages.

    Args:
        manifest_path: Path to dataset manifest CSV.
        raw_dir: Directory containing raw PDF files.

    Returns:
        List of all page records across all documents in the manifest.
    """
    manifest_data = load_manifest(manifest_path)
    validate_dataset(manifest_data, raw_dir)

    all_page_records = []
    for row in manifest_data:
        pdf_path = Path(raw_dir) / row["filename"]
        page_records = extract_pdf_pages(str(pdf_path), row)
        all_page_records.extend(page_records)

    return all_page_records


def main() -> None:
    """CLI entrypoint for corpus ingestion."""
    print("Starting document ingestion...")
    try:
        manifest_data = load_manifest()
        validate_dataset(manifest_data)
        records = ingest_corpus()

        # Calculate per-document summary
        doc_summary: Dict[str, Dict[str, Any]] = {}
        for r in records:
            doc_id = r["document_id"]
            if doc_id not in doc_summary:
                doc_summary[doc_id] = {
                    "filename": r["filename"],
                    "title": r["title"],
                    "pages": 0,
                    "char_count": 0,
                }
            doc_summary[doc_id]["pages"] += 1
            doc_summary[doc_id]["char_count"] += len(r["text"])

        print("\n--- Ingestion Summary ---")
        print(f"Total Documents Ingested: {len(doc_summary)}")
        print(f"Total Page Records: {len(records)}\n")
        print(f"{'Document ID':<35} {'Filename':<35} {'Pages':<6} {'Total Chars':<12}")
        print("-" * 90)

        for doc_id, info in doc_summary.items():
            print(f"{doc_id:<35} {info['filename']:<35} {info['pages']:<6} {info['char_count']:<12}")

        print("\nIngestion completed successfully!")

    except Exception as err:
        print(f"\n[ERROR] Ingestion failed: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
