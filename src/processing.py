"""Text processing module for Animal Knowledge RAG Assistant.

Applies light, non-destructive text normalization to extracted PDF pages
while preserving metadata, paragraph structure, punctuation, and scientific names.
"""

import re
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    """Apply light, non-destructive text normalization to raw extracted page text.

    Args:
        text: Raw page text.

    Returns:
        Normalized text string.

    Raises:
        ValueError: If input text is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty or whitespace-only.")

    # 1. Normalize line endings to standard Unix \n
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Line-by-line horizontal space collapse and line trimming
    lines = normalized.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces or tabs into a single space
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(cleaned)

    normalized = "\n".join(cleaned_lines)

    # 3. Normalize excessive blank lines (max 2 consecutive newlines to preserve paragraphs)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # 4. Final strip of leading/trailing blank lines
    normalized = normalized.strip()

    if not normalized:
        raise ValueError("Normalized text is empty or whitespace-only.")

    return normalized


def process_page(page_record: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single Stage 4 page record by normalizing its text.

    Args:
        page_record: Stage 4 page record dictionary.

    Returns:
        New dictionary with normalized text and preserved metadata.

    Raises:
        ValueError: If input page record is invalid or text normalization fails.
    """
    if "text" not in page_record:
        raise ValueError("Page record missing required 'text' field.")

    normalized_text = normalize_text(page_record["text"])

    # Create a new record to avoid mutating the original input
    return {
        "document_id": page_record["document_id"],
        "filename": page_record["filename"],
        "page_number": page_record["page_number"],
        "text": normalized_text,
        "title": page_record["title"],
        "publisher": page_record["publisher"],
        "source_url": page_record["source_url"],
    }


def process_pages(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a list of Stage 4 page records.

    Args:
        page_records: List of Stage 4 page record dictionaries.

    Returns:
        List of new page records containing normalized text.

    Raises:
        ValueError: If duplicate source page identities exist or any page fails processing.
    """
    seen_page_identities = set()
    processed_records = []

    for record in page_records:
        identity = (record["document_id"], record["page_number"])
        if identity in seen_page_identities:
            raise ValueError(
                f"Duplicate source page identity detected: document '{record['document_id']}', "
                f"page {record['page_number']}."
            )
        seen_page_identities.add(identity)

        processed = process_page(record)
        processed_records.append(processed)

    return processed_records
