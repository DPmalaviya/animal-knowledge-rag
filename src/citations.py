"""Stage 10 — Citations & Grounding module for Animal Knowledge RAG Assistant.

Implements deterministic citation extraction, fail-closed validation, trusted metadata lookup,
and user-facing filename/page rendering for grounded RAG answers.
"""

import argparse
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation import (
    DEFAULT_GENERATION_MODEL,
    INSUFFICIENT_CONTEXT_FALLBACK,
    rag_answer,
    validate_retrieval_records,
)

# Regex pattern for valid controlled citation groups: [C1], [C1, C2], [C1, C2, C4]
# Matches optional internal whitespace around commas and inside brackets.
VALID_CITATION_GROUP_PATTERN = re.compile(r"\[\s*C[1-9]\d*(?:\s*,\s*C[1-9]\d*)*\s*\]")

# Regex pattern for scanning suspicious/malformed citation-like bracket expressions.
# Targets bracketed strings that resemble controlled C-IDs (case variants, C0, punctuation errors, etc.)
# but fail strict grammar validation.
MALFORMED_CITATION_PATTERN = re.compile(
    r"\[\s*(?:"
    r"[cC]\d+|"  # Any c1, C0, C01, C1...
    r"C\d+\s*[\s;,]\s*|"  # C1; C2, C1 C2, C1, etc.
    r"[\s;,]\s*C\d+|"  # , C1, ; C1
    r"C\d+.*?,.*?"  # Multiple elements with commas
    r")\s*\]"
)


def validate_source_map(
    source_map: Dict[str, Dict[str, Any]],
    retrieved_context_count: Optional[int] = None,
) -> None:
    """Validate Stage 9 source_map dictionary integrity and provenance metadata.

    Args:
        source_map: Source map dictionary mapping C1..CK identifiers to retrieval records.
        retrieved_context_count: Expected context count matching len(source_map).

    Raises:
        ValueError: If source_map is invalid, non-canonical, non-contiguous, or fails retrieval provenance.
    """
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("source_map must be a non-empty dictionary.")

    k = len(source_map)
    if retrieved_context_count is not None:
        if isinstance(retrieved_context_count, bool) or not isinstance(retrieved_context_count, int):
            raise ValueError(f"retrieved_context_count must be an integer, got {type(retrieved_context_count)}.")
        if retrieved_context_count != k:
            raise ValueError(
                f"retrieved_context_count ({retrieved_context_count}) does not match source_map size ({k})."
            )

    expected_keys = [f"C{i}" for i in range(1, k + 1)]
    actual_keys = list(source_map.keys())

    if actual_keys != expected_keys:
        raise ValueError(
            f"source_map keys must be canonical contiguous C1..C{k}, got {actual_keys}."
        )

    # Validate provenance for each record in rank order
    ordered_records = []
    for i in range(1, k + 1):
        source_id = f"C{i}"
        rec = source_map[source_id]
        if not isinstance(rec, dict):
            raise ValueError(f"Record for '{source_id}' is not a dictionary.")
        rank = rec.get("rank")
        if rank != i:
            raise ValueError(f"Record for '{source_id}' has rank {rank}, expected {i}.")
        ordered_records.append(rec)

    # Delegate 12-field provenance validation to generation module
    validate_retrieval_records(ordered_records)


def validate_rag_result(rag_result: Dict[str, Any]) -> None:
    """Validate incoming Stage 9 result contract before citation resolution.

    Args:
        rag_result: Stage 9 result dictionary.

    Raises:
        ValueError: If required fields are missing, invalid, or malformed.
    """
    if not isinstance(rag_result, dict):
        raise ValueError("rag_result must be a dictionary.")

    question = rag_result.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("rag_result missing valid non-empty 'question' string.")

    answer = rag_result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("rag_result missing valid non-empty 'answer' string.")

    gen_model = rag_result.get("generation_model")
    if gen_model != DEFAULT_GENERATION_MODEL:
        raise ValueError(
            f"rag_result generation_model '{gen_model}' invalid. Expected '{DEFAULT_GENERATION_MODEL}'."
        )

    source_map = rag_result.get("source_map")
    context_count = rag_result.get("retrieved_context_count")

    validate_source_map(source_map, retrieved_context_count=context_count)


def parse_citation_groups(answer: str) -> List[Tuple[int, int, str, List[str]]]:
    """Parse valid controlled citation marker groups and detect malformed citation markers.

    Args:
        answer: Raw answer string from Stage 9 generation.

    Returns:
        List of tuples: (start_index, end_index, match_text, list_of_c_ids) for each valid group.

    Raises:
        ValueError: If malformed citation syntax, duplicate IDs within a group, or invalid C-IDs are found.
    """
    valid_matches: List[Tuple[int, int, str, List[str]]] = []
    valid_spans: Set[Tuple[int, int]] = set()

    for match in VALID_CITATION_GROUP_PATTERN.finditer(answer):
        start, end = match.span()
        raw_text = match.group(0)

        # Extract C-IDs inside brackets
        inner = raw_text[1:-1].strip()
        raw_ids = [token.strip() for token in inner.split(",")]

        # Check for non-canonical tokens or leading zeroes (e.g. C01, C0)
        c_ids = []
        for token in raw_ids:
            if not token.startswith("C") or not token[1:].isdigit():
                raise ValueError(f"Malformed controlled citation ID in group: '{token}' in '{raw_text}'.")
            num_str = token[1:]
            if num_str.startswith("0") or int(num_str) < 1:
                raise ValueError(f"Non-canonical controlled citation ID: '{token}' in '{raw_text}'.")
            c_ids.append(token)

        # Check for duplicate IDs within a single group [C1, C1]
        if len(c_ids) != len(set(c_ids)):
            raise ValueError(f"Duplicate citation ID found within single group '{raw_text}'.")

        valid_matches.append((start, end, raw_text, c_ids))
        valid_spans.add((start, end))

    # Scan for malformed citation-like bracket markers not captured by valid_spans
    for match in MALFORMED_CITATION_PATTERN.finditer(answer):
        span = match.span()
        if span not in valid_spans:
            # Verify if this match overlaps with any valid span
            overlaps = any(
                max(span[0], v_span[0]) < min(span[1], v_span[1])
                for v_span in valid_spans
            )
            if not overlaps:
                raise ValueError(
                    f"Malformed controlled citation marker detected: '{match.group(0)}'."
                )

    # Additional check: scan for lowercase [c1] or ambiguous brackets like [C1; C2], [C1 C2]
    # using a broad search for bracketed expressions starting with c/C
    for match in re.finditer(r"\[\s*[cC].*?\]", answer):
        span = match.span()
        if span not in valid_spans:
            overlaps = any(
                max(span[0], v_span[0]) < min(span[1], v_span[1])
                for v_span in valid_spans
            )
            if not overlaps:
                raise ValueError(
                    f"Malformed controlled citation marker detected: '{match.group(0)}'."
                )

    return valid_matches


def validate_citation_groups(
    parsed_groups: List[Tuple[int, int, str, List[str]]],
    source_map: Dict[str, Dict[str, Any]],
    answer: str,
) -> None:
    """Validate source ID existence and enforce fail-closed citation policies.

    Args:
        parsed_groups: List of parsed citation groups.
        source_map: Validated Stage 9 source map.
        answer: Raw answer string.

    Raises:
        ValueError: If unknown C-IDs are cited, or if supported answer has 0 citations,
                    or if fallback answer unexpectedly contains citations.
    """
    is_exact_fallback = answer.strip() == INSUFFICIENT_CONTEXT_FALLBACK.strip()
    contains_fallback_text = INSUFFICIENT_CONTEXT_FALLBACK in answer

    if contains_fallback_text:
        if parsed_groups:
            raise ValueError(
                "Fallback answer unexpectedly contains controlled citation markers."
            )
        if not is_exact_fallback:
            raise ValueError(
                "Fallback answer text contains unauthorized extra text."
            )
        return

    # Non-fallback answer: at least one valid citation group is required
    if not parsed_groups:
        raise ValueError(
            "Supported Stage 9 answer contains no valid controlled citation markers."
        )

    # Verify all cited source IDs exist in source_map
    valid_source_ids = set(source_map.keys())
    for _, _, match_text, c_ids in parsed_groups:
        for source_id in c_ids:
            if source_id not in valid_source_ids:
                raise ValueError(
                    f"Unknown citation source ID '{source_id}' in marker '{match_text}'."
                )


def format_inline_citation(
    c_ids: List[str],
    source_map: Dict[str, Dict[str, Any]],
) -> str:
    """Format inline filename and physical page reference for a group of C-IDs.

    Rules:
    - Group by filename, preserving first-appearance order of filenames.
    - Collect and deduplicate physical page numbers per filename.
    - Sort page numbers in ascending order.
    - Format single page as 'p. X' and multiple pages as 'pp. X, Y'.
    - Separate multiple filenames with '; '.

    Args:
        c_ids: List of source IDs in the citation group (e.g. ['C1', 'C2']).
        source_map: Validated source map.

    Returns:
        Formatted inline citation string (e.g. '(file.pdf, p. 1)' or '(file.pdf, pp. 1, 2)').
    """
    # Group physical pages by filename, preserving filename insertion order
    file_pages: Dict[str, List[int]] = {}
    for sid in c_ids:
        rec = source_map[sid]
        fn = rec["filename"]
        pg = rec["page_number"]
        if fn not in file_pages:
            file_pages[fn] = []
        if pg not in file_pages[fn]:
            file_pages[fn].append(pg)

    file_chunks: List[str] = []
    for fn, pages in file_pages.items():
        sorted_pages = sorted(pages)
        if len(sorted_pages) == 1:
            pg_str = f"p. {sorted_pages[0]}"
        else:
            pg_str = f"pp. {', '.join(str(p) for p in sorted_pages)}"
        file_chunks.append(f"{fn}, {pg_str}")

    return f"({'; '.join(file_chunks)})"


def render_answer_citations(
    raw_answer: str,
    parsed_groups: List[Tuple[int, int, str, List[str]]],
    source_map: Dict[str, Dict[str, Any]],
) -> str:
    """Replace raw [C#] citation markers with formatted inline filename/page citations.

    Preserves answer formatting, markdown bullets, and punctuation.

    Args:
        raw_answer: Original Stage 9 answer string.
        parsed_groups: Parsed citation groups.
        source_map: Validated source map.

    Returns:
        Rendered answer string with inline filename/page citations.
    """
    if not parsed_groups:
        return raw_answer

    # Replace from back to front to preserve string indices
    rendered = raw_answer
    for start, end, _, c_ids in reversed(parsed_groups):
        inline_text = format_inline_citation(c_ids, source_map)
        rendered = rendered[:start] + inline_text + rendered[end:]

    return rendered


def resolve_citation_record(
    source_id: str,
    source_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve source ID into a structured trusted citation metadata dictionary.

    Excludes raw embedding vectors, FAISS IDs, and similarity scores.

    Args:
        source_id: Canonical source ID (e.g. 'C1').
        source_map: Validated source map.

    Returns:
        Structured citation metadata dictionary.
    """
    rec = source_map[source_id]
    return {
        "source_id": source_id,
        "chunk_id": rec["chunk_id"],
        "document_id": rec["document_id"],
        "filename": rec["filename"],
        "page_number": rec["page_number"],
        "title": rec["title"],
        "publisher": rec["publisher"],
        "source_url": rec["source_url"],
        "rank": rec["rank"],
        "chunk_index": rec["chunk_index"],
    }


def resolve_rag_citations(rag_result: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Stage 10 deterministic citation resolution on a Stage 9 RAG result.

    Performs validation, marker parsing, fail-closed checking, trusted metadata lookup,
    and inline rendering without modifying the input rag_result.

    Operates 100% locally with 0 API / network calls.

    Args:
        rag_result: Return dictionary from Stage 9 rag_answer.

    Returns:
        Structured Stage 10 final result dictionary.

    Raises:
        ValueError: If rag_result, source_map, or citations fail validation.
    """
    # 1. Validate incoming Stage 9 result contract
    validate_rag_result(rag_result)

    question = rag_result["question"]
    raw_answer = rag_result["answer"]
    gen_model = rag_result["generation_model"]
    source_map = rag_result["source_map"]

    is_fallback = raw_answer.strip() == INSUFFICIENT_CONTEXT_FALLBACK.strip()

    # 2. Parse citation groups & detect malformed markers
    parsed_groups = parse_citation_groups(raw_answer)

    # 3. Validate citation groups against fail-closed rules
    validate_citation_groups(parsed_groups, source_map, raw_answer)

    # 4. Extract unique cited source IDs in first-appearance order
    citation_ids: List[str] = []
    seen_ids: Set[str] = set()
    for _, _, _, c_ids in parsed_groups:
        for sid in c_ids:
            if sid not in seen_ids:
                seen_ids.add(sid)
                citation_ids.append(sid)

    # 5. Build structured citation records
    citations = [
        resolve_citation_record(sid, source_map) for sid in citation_ids
    ]

    # 6. Render user-facing inline answer
    rendered_answer = render_answer_citations(raw_answer, parsed_groups, source_map)

    return {
        "question": question,
        "raw_answer": raw_answer,
        "rendered_answer": rendered_answer,
        "generation_model": gen_model,
        "citation_ids": citation_ids,
        "citations": citations,
        "citation_group_count": len(parsed_groups),
        "unique_citation_count": len(citation_ids),
        "is_fallback": is_fallback,
    }


def answer_with_citations(
    question: str,
    top_k: int = 4,
    index_dir: str = "index",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    model_name: str = DEFAULT_GENERATION_MODEL,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """End-to-end RAG convenience function (Stage 8 Retrieval + Stage 9 Generation + Stage 10 Citations).

    Executes retrieval and generation via rag_answer, then resolves citations deterministically.

    Args:
        question: User question string.
        top_k: Context chunk count (default 4).
        index_dir: FAISS index directory path.
        api_key: Optional Gemini API key.
        client: Optional pre-configured GenAI client.
        model_name: Generation model identifier.
        max_retries: Generation retries.
        retry_delay: Retry delay in seconds.

    Returns:
        Structured Stage 10 final result dictionary.
    """
    rag_res = rag_answer(
        question=question,
        top_k=top_k,
        index_dir=index_dir,
        api_key=api_key,
        client=client,
        model_name=model_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return resolve_rag_citations(rag_res)


def format_citation_cli(cited_result: Dict[str, Any]) -> str:
    """Format Stage 10 execution result for CLI display.

    Args:
        cited_result: Return dictionary from resolve_rag_citations or answer_with_citations.

    Returns:
        Formatted multi-line CLI display string.
    """
    lines = [
        "=" * 80,
        "STAGE 10 RAG RESULT WITH CITATIONS & GROUNDING",
        "=" * 80,
        f"QUESTION: {cited_result['question']}",
        f"GENERATION MODEL: {cited_result['generation_model']}",
        f"IS FALLBACK: {cited_result['is_fallback']}",
        f"CITATION GROUP COUNT: {cited_result['citation_group_count']}",
        f"UNIQUE CITATION COUNT: {cited_result['unique_citation_count']}",
        "-" * 80,
        "RAW ANSWER:",
        cited_result["raw_answer"],
        "-" * 80,
        "FINAL RENDERED ANSWER:",
        cited_result["rendered_answer"],
        "-" * 80,
        "CITATION DETAILS:",
    ]

    if not cited_result["citations"]:
        lines.append("  (No citations - fallback answer)")
    else:
        for cit in cited_result["citations"]:
            lines.extend([
                f"  [{cit['source_id']}]",
                f"    filename:   {cit['filename']}",
                f"    page:       {cit['page_number']}",
                f"    chunk_id:   {cit['chunk_id']}",
                f"    doc_id:     {cit['document_id']}",
                f"    title:      {cit['title']}",
                f"    publisher:  {cit['publisher']}",
                f"    source_url: {cit['source_url']}",
            ])

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for Stage 10 end-to-end RAG with citations execution."""
    parser = argparse.ArgumentParser(
        description="Animal Knowledge RAG Assistant — Stage 10 Citations & Grounding CLI"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Question string to answer using retrieved context.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        help="Number of top document chunks to retrieve for context (default: 4).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="index",
        help="Path to FAISS index directory (default: index).",
    )

    args = parser.parse_args()

    print(f"Executing Stage 10 RAG Pipeline for query: \"{args.query}\"...")
    try:
        result = answer_with_citations(
            question=args.query,
            top_k=args.top_k,
            index_dir=args.index_dir,
        )
        print("\n" + format_citation_cli(result))
        print("Stage 10 Citations & Grounding completed successfully!")
    except Exception as e:
        print(f"\nERROR: Stage 10 Citation Resolution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
