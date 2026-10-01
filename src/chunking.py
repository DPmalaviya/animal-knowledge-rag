"""Page-aware and paragraph-aware chunking module for Animal Knowledge RAG Assistant.

Splits processed page records into deterministic, page-bounded chunks using
paragraph and sentence boundaries, bounded overlap, and hard maximum size limits.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.ingestion import ingest_corpus
from src.processing import process_pages


# Size constants (in estimated tokens, ~4 characters per token)
TARGET_TOKENS = 450
OVERLAP_TOKENS = 75
HARD_MAX_TOKENS = 600
SMALL_CHUNK_THRESHOLD = 100


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using deterministic length heuristic.

    Formula: max(1, len(text) // 4)

    Args:
        text: Input text string.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    """Split a paragraph into sentences using standard-library regex heuristic.

    Args:
        paragraph: Paragraph text block.

    Returns:
        List of non-empty sentence strings.
    """
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sentences if s.strip()]


def split_sentence_into_words(sentence: str) -> List[str]:
    """Split a sentence into words for fallback splitting of oversized sentences.

    Args:
        sentence: Sentence string.

    Returns:
        List of non-empty word strings.
    """
    words = sentence.split(" ")
    return [w for w in words if w]


def _slice_oversized_string(text: str, max_tokens: int) -> List[str]:
    """Forcibly slice unusually long uninterrupted strings (e.g. URLs) into sub-blocks.

    Args:
        text: Uninterrupted long text string.
        max_tokens: Maximum allowed tokens per slice.

    Returns:
        List of sliced text blocks.
    """
    max_chars = max(4, max_tokens * 4)
    slices = []
    for i in range(0, len(text), max_chars):
        slices.append(text[i : i + max_chars])
    return slices


def _get_atomic_blocks(text: str) -> List[str]:
    """Break page text into atomic blocks (paragraphs, sentences, or word slices).

    Ensures no single block exceeds HARD_MAX_TOKENS.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    blocks = []

    for para in paragraphs:
        if estimate_tokens(para) <= TARGET_TOKENS:
            blocks.append(para)
        else:
            sentences = split_paragraph_into_sentences(para)
            if not sentences:
                sentences = [para]

            for sent in sentences:
                if estimate_tokens(sent) <= HARD_MAX_TOKENS:
                    blocks.append(sent)
                else:
                    words = split_sentence_into_words(sent)
                    if not words:
                        words = [sent]

                    current_word_block = []
                    current_word_tokens = 0
                    for word in words:
                        word_tok = estimate_tokens(word)
                        if word_tok > HARD_MAX_TOKENS:
                            if current_word_block:
                                blocks.append(" ".join(current_word_block))
                                current_word_block = []
                                current_word_tokens = 0
                            slices = _slice_oversized_string(word, HARD_MAX_TOKENS)
                            blocks.extend(slices)
                        elif current_word_tokens + word_tok + 1 > HARD_MAX_TOKENS:
                            if current_word_block:
                                blocks.append(" ".join(current_word_block))
                            current_word_block = [word]
                            current_word_tokens = word_tok
                        else:
                            current_word_block.append(word)
                            current_word_tokens += word_tok + 1

                    if current_word_block:
                        blocks.append(" ".join(current_word_block))

    return blocks


def _compute_overlap_text(prev_chunk_text: str, max_overlap_tokens: int) -> str:
    """Extract trailing overlap text from previous chunk up to max_overlap_tokens.

    Prefers whole sentence boundaries when available.
    """
    if not prev_chunk_text or max_overlap_tokens <= 0:
        return ""

    sentences = split_paragraph_into_sentences(prev_chunk_text)
    if not sentences:
        sentences = [prev_chunk_text]

    overlap_sentences = []
    accumulated_tokens = 0

    for sent in reversed(sentences):
        sent_tokens = estimate_tokens(sent)
        if accumulated_tokens + sent_tokens > max_overlap_tokens:
            if not overlap_sentences:
                # Fall back to word-level overlap if single sentence exceeds overlap budget
                words = split_sentence_into_words(sent)
                overlap_words = []
                w_tokens = 0
                for w in reversed(words):
                    wt = estimate_tokens(w)
                    if w_tokens + wt > max_overlap_tokens:
                        break
                    overlap_words.insert(0, w)
                    w_tokens += wt
                return " ".join(overlap_words)
            break

        overlap_sentences.insert(0, sent)
        accumulated_tokens += sent_tokens

    return " ".join(overlap_sentences)


def chunk_processed_page(page_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunk a single processed page record into page-bounded chunk records.

    Args:
        page_record: Processed page record dictionary.

    Returns:
        List of chunk record dictionaries strictly belonging to this physical page.
    """
    text = page_record["text"]
    doc_id = page_record["document_id"]
    filename = page_record["filename"]
    page_num = page_record["page_number"]
    title = page_record["title"]
    publisher = page_record["publisher"]
    source_url = page_record["source_url"]

    atomic_blocks = _get_atomic_blocks(text)
    if not atomic_blocks:
        raise ValueError(
            f"No valid text blocks found for document '{doc_id}' page {page_num}."
        )

    raw_chunks_text: List[str] = []
    current_blocks: List[str] = []
    current_tokens = 0

    for block in atomic_blocks:
        block_tokens = estimate_tokens(block)

        # Check if adding this block exceeds target or hard max
        if current_blocks and (current_tokens + block_tokens > TARGET_TOKENS or current_tokens + block_tokens > HARD_MAX_TOKENS):
            # Finalize current chunk text
            chunk_str = "\n\n".join(current_blocks)
            raw_chunks_text.append(chunk_str)

            # Compute overlap from finalized chunk
            overlap_str = _compute_overlap_text(chunk_str, OVERLAP_TOKENS)

            if overlap_str and estimate_tokens(overlap_str + "\n\n" + block) <= HARD_MAX_TOKENS:
                current_blocks = [overlap_str, block]
                current_tokens = estimate_tokens(overlap_str) + block_tokens
            else:
                current_blocks = [block]
                current_tokens = block_tokens
        else:
            current_blocks.append(block)
            current_tokens += block_tokens

    if current_blocks:
        raw_chunks_text.append("\n\n".join(current_blocks))

    # Small chunk merging within the same page
    if len(raw_chunks_text) > 1:
        last_chunk_tokens = estimate_tokens(raw_chunks_text[-1])
        if last_chunk_tokens < SMALL_CHUNK_THRESHOLD:
            prev_tokens = estimate_tokens(raw_chunks_text[-2])
            merged_text = raw_chunks_text[-2] + "\n\n" + raw_chunks_text[-1]
            if estimate_tokens(merged_text) <= HARD_MAX_TOKENS:
                raw_chunks_text[-2] = merged_text
                raw_chunks_text.pop()

    # Construct final chunk records
    chunk_records = []
    for idx, c_text in enumerate(raw_chunks_text, start=1):
        chunk_id = f"{doc_id}_p{page_num:03d}_c{idx:03d}"
        record = {
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "filename": filename,
            "page_number": page_num,
            "chunk_index": idx,
            "text": c_text,
            "title": title,
            "publisher": publisher,
            "source_url": source_url,
            "estimated_token_count": estimate_tokens(c_text),
        }
        chunk_records.append(record)

    return chunk_records


def chunk_pages(processed_page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk all processed page records into a flat collection of chunk records.

    Args:
        processed_page_records: List of processed page record dictionaries.

    Returns:
        List of all chunk record dictionaries across the corpus.

    Raises:
        ValueError: If duplicate chunk IDs or unrepresented source pages are detected.
    """
    all_chunks = []
    seen_chunk_ids = set()
    seen_source_pages = set()

    for page_record in processed_page_records:
        page_id = (page_record["document_id"], page_record["page_number"])
        seen_source_pages.add(page_id)

        chunks = chunk_processed_page(page_record)
        for c in chunks:
            cid = c["chunk_id"]
            if cid in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id generated: '{cid}'")
            seen_chunk_ids.add(cid)
            all_chunks.append(c)

    # Verification: Every input page identity must be represented
    input_pages = {(p["document_id"], p["page_number"]) for p in processed_page_records}
    if seen_source_pages != input_pages:
        missing = input_pages - seen_source_pages
        raise ValueError(f"Source pages missing from chunk output: {missing}")

    return all_chunks


def main() -> None:
    """CLI entrypoint for executing Stage 5 chunking pipeline and printing stats."""
    print("Executing Stage 5 Pipeline: Ingestion -> Processing -> Chunking...")

    # 1. Ingestion
    raw_pages = ingest_corpus()
    raw_chars = sum(len(p["text"]) for p in raw_pages)

    # 2. Processing
    processed_pages = process_pages(raw_pages)
    proc_chars = sum(len(p["text"]) for p in processed_pages)

    # 3. Chunking
    chunks = chunk_pages(processed_pages)

    # Statistical Calculations
    doc_ids = {c["document_id"] for c in chunks}
    source_page_ids = {(c["document_id"], c["page_number"]) for c in chunks}

    sizes = [c["estimated_token_count"] for c in chunks]
    min_size = min(sizes)
    max_size = max(sizes)
    mean_size = sum(sizes) / len(sizes)
    sorted_sizes = sorted(sizes)
    mid = len(sorted_sizes) // 2
    median_size = (sorted_sizes[mid] if len(sorted_sizes) % 2 != 0
                   else (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2)

    small_chunks = [c for c in chunks if c["estimated_token_count"] < SMALL_CHUNK_THRESHOLD]
    chunks_over_target = [c for c in chunks if c["estimated_token_count"] > TARGET_TOKENS]
    chunks_over_hard_max = [c for c in chunks if c["estimated_token_count"] > HARD_MAX_TOKENS]
    empty_chunks = [c for c in chunks if not c["text"].strip()]

    # Per-document aggregation
    doc_stats: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        d_id = c["document_id"]
        if d_id not in doc_stats:
            doc_stats[d_id] = {
                "filename": c["filename"],
                "pages": set(),
                "chunk_count": 0,
                "token_sum": 0,
            }
        doc_stats[d_id]["pages"].add(c["page_number"])
        doc_stats[d_id]["chunk_count"] += 1
        doc_stats[d_id]["token_sum"] += c["estimated_token_count"]

    print("\n" + "=" * 80)
    print("STAGE 5 CORPUS CHUNKING SUMMARY")
    print("=" * 80)
    print(f"Input Documents:          {len(set(p['document_id'] for p in raw_pages))}")
    print(f"Input Source Pages:       {len(raw_pages)}")
    print(f"Raw Input Characters:     {raw_chars:,}")
    print(f"Processed Page Chars:     {proc_chars:,}")
    print(f"Total Chunks Generated:   {len(chunks)}")
    print(f"Documents Represented:    {len(doc_ids)}")
    print(f"Source Pages Represented: {len(source_page_ids)}")
    print(f"Empty Chunks:             {len(empty_chunks)}")
    print(f"Duplicate Chunk IDs:      0")
    print("-" * 80)
    print("CHUNK SIZE METRICS (in estimated tokens, ~4 chars/token):")
    print(f"Target Size:              {TARGET_TOKENS} tokens")
    print(f"Hard Maximum Size:        {HARD_MAX_TOKENS} tokens")
    print(f"Minimum Size:             {min_size} tokens")
    print(f"Maximum Size:             {max_size} tokens")
    print(f"Mean Size:                {mean_size:.1f} tokens")
    print(f"Median Size:              {median_size:.1f} tokens")
    print(f"Small Chunks (<{SMALL_CHUNK_THRESHOLD} tok):  {len(small_chunks)}")
    print(f"Chunks Exceeding Target:  {len(chunks_over_target)}")
    print(f"Chunks Exceeding HardMax: {len(chunks_over_hard_max)}")
    print("=" * 80)

    print(f"\n{'Document ID':<35} {'Filename':<35} {'Pages':<6} {'Chunks':<7}")
    print("-" * 85)
    for d_id, stat in doc_stats.items():
        print(f"{d_id:<35} {stat['filename']:<35} {len(stat['pages']):<6} {stat['chunk_count']:<7}")

    # Optional: Save intermediate chunks JSON summary under ignored data/processed/
    proc_dir = Path("data/processed")
    if proc_dir.is_dir():
        summary_file = proc_dir / "chunks_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"\nSaved deterministic chunks snapshot to: {summary_file}")

    print("\nStage 5 pipeline completed successfully!")


if __name__ == "__main__":
    main()
