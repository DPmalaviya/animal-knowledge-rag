"""Deterministic, dependency-free lexical retrieval over local chunk metadata."""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

_REQUIRED_FIELDS = frozenset(
    {
        "faiss_id",
        "chunk_id",
        "document_id",
        "filename",
        "page_number",
        "chunk_index",
        "text",
        "title",
        "publisher",
        "source_url",
        "estimated_token_count",
        "embedding_model",
        "embedding_dimension",
    }
)


def tokenize_meaningful(text: str) -> List[str]:
    """Return lowercase ASCII-alphanumeric terms after fixed local filtering."""
    if not isinstance(text, str):
        raise ValueError("Text to tokenize must be a string.")

    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOP_WORDS and (len(token) >= 3 or token.isdigit())
    ]


def split_sentences(text: str) -> List[str]:
    """Split text after sentence punctuation while preserving source wording."""
    if not isinstance(text, str):
        raise ValueError("Text to split must be a string.")
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _sentence_dedup_key(sentence: str) -> str:
    return _WHITESPACE_RE.sub(" ", sentence).lower()


def _contains_controlled_citation_syntax(sentence: str) -> bool:
    from src.citations import parse_citation_groups

    try:
        return bool(parse_citation_groups(sentence))
    except ValueError:
        return True


def build_offline_fallback_result(question: str) -> Dict[str, Any]:
    """Return the final citation-stage shape for unsupported offline questions."""
    from src.generation import DEFAULT_GENERATION_MODEL, INSUFFICIENT_CONTEXT_FALLBACK

    _validate_question(question)
    return {
        "question": question,
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


def build_extractive_rag_result(
    question: str, retrieval_records: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Build a deterministic Stage 9 result from verbatim supported sentences."""
    from src.generation import DEFAULT_GENERATION_MODEL, build_source_map

    _validate_question(question)
    if not isinstance(retrieval_records, list):
        raise ValueError("Retrieval records must be a list.")
    if not retrieval_records:
        return None

    source_map = build_source_map(retrieval_records)
    query_terms = set(tokenize_meaningful(question))
    if not query_terms:
        return None
    minimum_distinct_overlap = (
        1 if len(query_terms) == 1 else max(2, math.ceil(len(query_terms) / 2))
    )

    candidates: List[Tuple[int, int, int, str, str]] = []
    for record_index, record in enumerate(retrieval_records):
        source_id = f"C{record_index + 1}"
        for sentence_position, sentence in enumerate(split_sentences(record["text"])):
            counts = Counter(tokenize_meaningful(sentence))
            overlap = query_terms.intersection(counts)
            if len(overlap) < minimum_distinct_overlap:
                continue
            if _contains_controlled_citation_syntax(sentence):
                continue
            score = sum(counts[term] for term in overlap)
            candidates.append(
                (score, record["rank"], sentence_position, sentence, source_id)
            )

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: List[Tuple[str, str]] = []
    seen = set()
    selected_character_count = 0
    for _, _, _, sentence, source_id in candidates:
        dedup_key = _sentence_dedup_key(sentence)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        if selected_character_count + len(sentence) > 600:
            continue
        selected.append((sentence, source_id))
        selected_character_count += len(sentence)
        if len(selected) == 3:
            break

    if not selected:
        return None

    answer = "\n".join(
        f"{sentence} [{source_id}]" for sentence, source_id in selected
    )
    return {
        "question": question,
        "answer": answer,
        "generation_model": DEFAULT_GENERATION_MODEL,
        "source_map": source_map,
        "retrieved_context_count": len(source_map),
        "usage_metadata": None,
        "answer_mode": "offline_extractive",
    }


def resolve_extractive_result(
    question: str, retrieval_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build and resolve an extractive result, or return the offline fallback."""
    from src.citations import resolve_rag_citations

    rag_result = build_extractive_rag_result(question, retrieval_records)
    if rag_result is None:
        return build_offline_fallback_result(question)
    resolved = resolve_rag_citations(rag_result)
    resolved["answer_mode"] = "offline_extractive"
    return resolved


def _validate_question(question: Any) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    return question


def _validate_top_k(top_k: Any, available: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if top_k > available:
        raise ValueError(
            f"top_k ({top_k}) exceeds available metadata records ({available})."
        )
    return top_k


def _validate_record(record: Any, position: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Metadata record at index {position} must be a dictionary.")
    missing = _REQUIRED_FIELDS.difference(record)
    if missing:
        raise ValueError(
            f"Metadata record at index {position} is missing fields: {sorted(missing)}"
        )

    faiss_id = record["faiss_id"]
    if isinstance(faiss_id, bool) or not isinstance(faiss_id, int) or faiss_id < 0:
        raise ValueError(f"Metadata record at index {position} has invalid faiss_id.")

    for field in (
        "chunk_id",
        "document_id",
        "filename",
        "text",
        "title",
        "publisher",
        "source_url",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(
                f"Metadata record at index {position} has invalid {field}."
            )

    for field in ("page_number", "chunk_index", "estimated_token_count"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Metadata record at index {position} has invalid {field}."
            )

    if record["embedding_model"] != "gemini-embedding-2":
        raise ValueError(
            f"Metadata record at index {position} has invalid embedding_model."
        )
    dimension = record["embedding_dimension"]
    if isinstance(dimension, bool) or dimension != 768:
        raise ValueError(
            f"Metadata record at index {position} has invalid embedding_dimension."
        )


def _validate_metadata(metadata: Any) -> List[Dict[str, Any]]:
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Metadata must be a non-empty list.")

    seen_ids = set()
    seen_chunk_ids = set()
    for position, record in enumerate(metadata):
        _validate_record(record, position)
        faiss_id = record["faiss_id"]
        if faiss_id in seen_ids:
            raise ValueError(f"Duplicate faiss_id {faiss_id} in metadata.")
        if faiss_id != position:
            raise ValueError(
                "Metadata faiss_id values must match canonical metadata-list order."
            )
        chunk_id = record["chunk_id"]
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"duplicate chunk_id '{chunk_id}' in metadata.")
        seen_ids.add(faiss_id)
        seen_chunk_ids.add(chunk_id)
    return metadata


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _chunk_ids_fingerprint(metadata: List[Dict[str, Any]]) -> str:
    chunk_ids = [record["chunk_id"] for record in metadata]
    serialized = json.dumps(chunk_ids, separators=(",", ":"))
    return _sha256_bytes(serialized.encode("utf-8"))


def _validate_manifest(
    manifest: Any, metadata: List[Dict[str, Any]], metadata_sha256: str
) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("Loaded manifest is not a dictionary.")
    if manifest.get("metadata_count") != len(metadata):
        raise ValueError("Manifest metadata_count does not match metadata length.")
    if manifest.get("embedding_model") != "gemini-embedding-2":
        raise ValueError("Manifest embedding_model does not match expected model.")
    dimension = manifest.get("dimension")
    if isinstance(dimension, bool) or dimension != 768:
        raise ValueError("Manifest dimension does not match expected dimension.")
    if manifest.get("chunk_metadata_sha256") != metadata_sha256:
        raise ValueError("Metadata checksum mismatch.")
    if manifest.get("ordered_chunk_ids_sha256") != _chunk_ids_fingerprint(metadata):
        raise ValueError("Ordered chunk IDs fingerprint mismatch.")


def rank_metadata(
    question: str, metadata: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """Rank metadata by normalized lexical score and canonical stored-list order.

    The normalization denominator counts every unweighted meaningful token
    occurrence in the candidate's title plus text. Because validated ``faiss_id``
    values equal list positions, ascending ``faiss_id`` is canonical stored order.
    """
    _validate_question(question)
    validated_metadata = _validate_metadata(metadata)
    _validate_top_k(top_k, len(validated_metadata))

    query_terms = set(tokenize_meaningful(question))
    if not query_terms:
        return []

    scored = []
    for record in validated_metadata:
        title_tokens = tokenize_meaningful(record["title"])
        text_tokens = tokenize_meaningful(record["text"])
        title_counts = Counter(title_tokens)
        text_counts = Counter(text_tokens)
        numerator = sum(
            title_counts[term] * 3 + text_counts[term] for term in query_terms
        )
        if numerator == 0:
            continue

        # Denominator counts unweighted title + text token occurrences.
        candidate_token_count = len(title_tokens) + len(text_tokens)
        score = numerator / math.sqrt(candidate_token_count)
        result = dict(record)
        result["similarity_score"] = float(score)
        scored.append(result)

    # faiss_id equals canonical stored list position after metadata validation.
    scored.sort(key=lambda record: (-record["similarity_score"], record["faiss_id"]))
    results = scored[:top_k]
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def retrieve_locally(
    question: str, index_dir: str = "index", top_k: int = 4
) -> List[Dict[str, Any]]:
    """Load local chunk metadata and return deterministic lexical matches."""
    _validate_question(question)
    metadata_path = Path(index_dir) / "chunk_metadata.json"
    manifest_path = Path(index_dir) / "index_manifest.json"
    try:
        metadata_bytes = metadata_path.read_bytes()
    except (OSError, TypeError) as exc:
        raise ValueError(f"Unable to read metadata file: {metadata_path}") from exc

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Malformed metadata JSON: {metadata_path}") from exc

    validated_metadata = _validate_metadata(metadata)
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except (OSError, TypeError) as exc:
        raise ValueError(f"Unable to read manifest file: {manifest_path}") from exc
    try:
        manifest = json.loads(manifest_text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Malformed manifest JSON: {manifest_path}") from exc

    _validate_manifest(manifest, validated_metadata, _sha256_bytes(metadata_bytes))
    _validate_top_k(top_k, len(validated_metadata))
    return rank_metadata(question, validated_metadata, top_k)
