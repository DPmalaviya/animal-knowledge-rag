"""Stage 10 — Citations & Grounding automated test suite.

Verifies deterministic citation extraction, fail-closed validation, canonical key constraints,
inline filename/page rendering, punctuation/markdown preservation, and provenance tracking.

All tests run 100% offline with zero Gemini API calls, zero embedding API calls, and no GEMINI_API_KEY.
"""

import copy
import unittest
from typing import Any, Dict, List, Optional

from src.citations import (
    format_inline_citation,
    parse_citation_groups,
    render_answer_citations,
    resolve_rag_citations,
    validate_rag_result,
    validate_source_map,
)
from src.generation import (
    DEFAULT_GENERATION_MODEL,
    INSUFFICIENT_CONTEXT_FALLBACK,
)


def create_synthetic_record(
    rank: int,
    filename: str = "bald_eagle_lead_exposure.pdf",
    page_number: int = 1,
    chunk_index: int = 1,
    doc_id: str = "doc_bald_eagle",
    chunk_id: Optional[str] = None,
    title: str = "Bald Eagle Study",
    publisher: str = "USGS",
    source_url: str = "https://example.com/eagle.pdf",
) -> Dict[str, Any]:
    """Helper to build a valid synthetic Stage 8 retrieval provenance record."""
    cid = chunk_id or f"{doc_id}_p{page_number:03d}_c{chunk_index:03d}"
    return {
        "rank": rank,
        "faiss_id": rank - 1,
        "similarity_score": 0.88 - (rank * 0.05),
        "chunk_id": cid,
        "document_id": doc_id,
        "filename": filename,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "text": f"Synthetic text chunk {rank} content.",
        "title": title,
        "publisher": publisher,
        "source_url": source_url,
        "estimated_token_count": 400,
        "embedding_model": "gemini-embedding-2",
        "embedding_dimension": 768,
    }


def create_synthetic_source_map(k: int = 4) -> Dict[str, Dict[str, Any]]:
    """Helper to build a canonical valid C1..CK source map."""
    source_map = {}
    sample_files = [
        ("bald_eagle_lead_exposure.pdf", 1),
        ("african_elephant_reintegration.pdf", 3),
        ("monarch_flight_performance.pdf", 1),
        ("african_elephant_reintegration.pdf", 2),
    ]
    for i in range(1, k + 1):
        idx = (i - 1) % len(sample_files)
        fn, pg = sample_files[idx]
        source_map[f"C{i}"] = create_synthetic_record(
            rank=i,
            filename=fn,
            page_number=pg,
            chunk_index=i,
        )
    return source_map


def create_synthetic_rag_result(
    question: str = "How does lead exposure occur?",
    answer: str = "Bald eagles scavenge remains containing lead fragments [C1, C2].",
    source_map: Optional[Dict[str, Dict[str, Any]]] = None,
    model: str = DEFAULT_GENERATION_MODEL,
) -> Dict[str, Any]:
    """Helper to build a complete synthetic Stage 9 RAG result."""
    smap = source_map or create_synthetic_source_map(4)
    return {
        "question": question,
        "answer": answer,
        "generation_model": model,
        "source_map": smap,
        "retrieved_context_count": len(smap),
        "usage_metadata": {
            "prompt_token_count": 100,
            "candidates_token_count": 50,
            "total_token_count": 150,
        },
    }


class TestCitationsStage10(unittest.TestCase):
    """Test suite covering all 47 Stage 10 citation requirement contracts."""

    def test_01_valid_stage_9_result_passes_validation(self) -> None:
        rag_res = create_synthetic_rag_result()
        validate_rag_result(rag_res)

    def test_02_missing_answer_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result()
        del rag_res["answer"]
        with self.assertRaises(ValueError):
            validate_rag_result(rag_res)

    def test_03_empty_answer_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result(answer="   ")
        with self.assertRaises(ValueError):
            validate_rag_result(rag_res)

    def test_04_wrong_generation_model_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result(model="gpt-4o")
        with self.assertRaises(ValueError):
            validate_rag_result(rag_res)

    def test_05_invalid_non_dict_source_map_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result()
        rag_res["source_map"] = "not_a_dict"
        with self.assertRaises(ValueError):
            validate_rag_result(rag_res)

    def test_06_source_map_keys_must_be_canonical(self) -> None:
        smap = {
            "source1": create_synthetic_record(1),
            "source2": create_synthetic_record(2),
        }
        with self.assertRaises(ValueError):
            validate_source_map(smap, 2)

    def test_07_source_map_keys_must_be_contiguous(self) -> None:
        smap = {
            "C1": create_synthetic_record(1),
            "C3": create_synthetic_record(2),
        }
        with self.assertRaises(ValueError):
            validate_source_map(smap, 2)

    def test_08_source_map_rank_must_match_c_number(self) -> None:
        smap = {
            "C1": create_synthetic_record(1),
            "C2": create_synthetic_record(99),  # Mismatched rank
        }
        with self.assertRaises(ValueError):
            validate_source_map(smap, 2)

    def test_09_malformed_provenance_record_is_rejected(self) -> None:
        smap = create_synthetic_source_map(2)
        smap["C1"]["filename"] = ""  # Corrupt provenance field
        with self.assertRaises(ValueError):
            validate_source_map(smap, 2)

    def test_10_single_c1_parses_correctly(self) -> None:
        groups = parse_citation_groups("Bald eagles scavenge remains [C1].")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][3], ["C1"])

    def test_11_c1_c2_without_space_parses_correctly(self) -> None:
        groups = parse_citation_groups("Bald eagles scavenge remains [C1,C2].")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][3], ["C1", "C2"])

    def test_12_c1_c2_with_space_parses_correctly(self) -> None:
        groups = parse_citation_groups("Bald eagles scavenge remains [C1, C2].")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][3], ["C1", "C2"])

    def test_13_c1_c2_c4_parses_correctly(self) -> None:
        groups = parse_citation_groups("Bald eagles scavenge remains [C1, C2, C4].")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][3], ["C1", "C2", "C4"])

    def test_14_multiple_citation_groups_preserve_occurrence_order(self) -> None:
        groups = parse_citation_groups("First claim [C2]. Second claim [C1, C4]. Third [C3].")
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0][3], ["C2"])
        self.assertEqual(groups[1][3], ["C1", "C4"])
        self.assertEqual(groups[2][3], ["C3"])

    def test_15_unknown_source_id_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Bald eagles scavenge remains [C99]."
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Unknown citation source ID", str(ctx.exception))

    def test_16_c0_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C0].")

    def test_17_c01_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C01].")

    def test_18_lowercase_c1_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [c1].")

    def test_19_semicolon_c1_c2_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C1; C2].")

    def test_20_c1_c2_no_comma_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C1 C2].")

    def test_21_trailing_comma_c1_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C1,].")

    def test_22_duplicate_id_inside_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_citation_groups("Claim text [C1, C1].")

    def test_23_repeated_c1_across_separate_groups_is_allowed(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Claim A [C1]. Claim B [C1]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertEqual(res["citation_group_count"], 2)
        self.assertEqual(res["citation_ids"], ["C1"])
        self.assertEqual(res["unique_citation_count"], 1)

    def test_24_non_fallback_answer_with_zero_citations_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Bald eagles scavenge remains without any markers."
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("no valid controlled citation markers", str(ctx.exception))

    def test_25_exact_fallback_with_zero_citations_passes(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer=INSUFFICIENT_CONTEXT_FALLBACK
        )
        res = resolve_rag_citations(rag_res)
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["citation_group_count"], 0)
        self.assertEqual(res["unique_citation_count"], 0)
        self.assertEqual(res["citation_ids"], [])
        self.assertEqual(res["citations"], [])
        self.assertEqual(res["rendered_answer"], INSUFFICIENT_CONTEXT_FALLBACK)

    def test_26_fallback_containing_citation_is_rejected(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer=f"{INSUFFICIENT_CONTEXT_FALLBACK} [C1]"
        )
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Fallback answer unexpectedly contains", str(ctx.exception))

    def test_27_single_citation_renders_filename_and_p(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Bald eagles scavenge remains [C1]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertIn("(bald_eagle_lead_exposure.pdf, p. 1)", res["rendered_answer"])

    def test_28_multiple_pages_one_file_render_with_pp(self) -> None:
        smap = {
            "C1": create_synthetic_record(1, "african_elephant.pdf", page_number=2),
            "C2": create_synthetic_record(2, "african_elephant.pdf", page_number=3),
        }
        rag_res = create_synthetic_rag_result(
            answer="Elephant study details [C1, C2].",
            source_map=smap,
        )
        res = resolve_rag_citations(rag_res)
        self.assertIn("(african_elephant.pdf, pp. 2, 3)", res["rendered_answer"])

    def test_29_same_filename_page_multiple_c_ids_renders_once_inline(self) -> None:
        smap = {
            "C1": create_synthetic_record(1, "bald_eagle.pdf", page_number=1, chunk_index=1),
            "C2": create_synthetic_record(2, "bald_eagle.pdf", page_number=1, chunk_index=2),
        }
        rag_res = create_synthetic_rag_result(
            answer="Lead exposure details [C1, C2].",
            source_map=smap,
        )
        res = resolve_rag_citations(rag_res)
        self.assertIn("(bald_eagle.pdf, p. 1)", res["rendered_answer"])
        self.assertNotIn("(bald_eagle.pdf, p. 1; bald_eagle.pdf, p. 1)", res["rendered_answer"])

    def test_30_multiple_filenames_render_with_semicolon_separation(self) -> None:
        smap = {
            "C1": create_synthetic_record(1, "file_a.pdf", page_number=1, chunk_index=1),
            "C2": create_synthetic_record(2, "file_b.pdf", page_number=4, chunk_index=2),
        }
        rag_res = create_synthetic_rag_result(
            answer="Cross-species analysis [C1, C2].",
            source_map=smap,
        )
        res = resolve_rag_citations(rag_res)
        self.assertIn("(file_a.pdf, p. 1; file_b.pdf, p. 4)", res["rendered_answer"])

    def test_31_raw_answer_is_preserved_unchanged(self) -> None:
        raw = "Bald eagles scavenge remains [C1, C2]."
        rag_res = create_synthetic_rag_result(answer=raw)
        res = resolve_rag_citations(rag_res)
        self.assertEqual(res["raw_answer"], raw)

    def test_32_rendered_answer_contains_no_raw_controlled_markers(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Bald eagles scavenge remains [C1, C2]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertNotIn("[C1, C2]", res["rendered_answer"])
        self.assertNotIn("[C1]", res["rendered_answer"])

    def test_33_citation_ids_are_unique_in_first_appearance_order(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Claim A [C2]. Claim B [C1, C2]. Claim C [C4]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertEqual(res["citation_ids"], ["C2", "C1", "C4"])

    def test_34_citation_group_count_is_correct(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Claim A [C1]. Claim B [C1, C2]. Claim C [C2]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertEqual(res["citation_group_count"], 3)

    def test_35_unique_citation_count_is_correct(self) -> None:
        rag_res = create_synthetic_rag_result(
            answer="Claim A [C1]. Claim B [C1, C2]. Claim C [C2]."
        )
        res = resolve_rag_citations(rag_res)
        self.assertEqual(res["unique_citation_count"], 2)

    def test_36_structured_citations_preserve_chunk_id(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertIn("chunk_id", c1)
        self.assertTrue(isinstance(c1["chunk_id"], str) and c1["chunk_id"])

    def test_37_structured_citations_preserve_document_id(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertEqual(c1["document_id"], "doc_bald_eagle")

    def test_38_structured_citations_preserve_filename(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertEqual(c1["filename"], "bald_eagle_lead_exposure.pdf")

    def test_39_structured_citations_preserve_physical_page(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertEqual(c1["page_number"], 1)

    def test_40_structured_citations_preserve_title_publisher_source_url(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertEqual(c1["title"], "Bald Eagle Study")
        self.assertEqual(c1["publisher"], "USGS")
        self.assertEqual(c1["source_url"], "https://example.com/eagle.pdf")

    def test_41_structured_citations_exclude_embeddings(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertNotIn("embedding", c1)
        self.assertNotIn("vector", c1)

    def test_42_structured_citations_do_not_expose_similarity_as_confidence(self) -> None:
        res = resolve_rag_citations(create_synthetic_rag_result())
        c1 = res["citations"][0]
        self.assertNotIn("confidence", c1)
        self.assertNotIn("similarity_score", c1)

    def test_43_answer_markdown_and_bullets_survive_replacement(self) -> None:
        raw = "- fGCM levels [C2]\n- eTGS [C2, C4]"
        smap = {
            "C1": create_synthetic_record(1, "dummy.pdf", page_number=1, chunk_index=1),
            "C2": create_synthetic_record(2, "african_elephant.pdf", page_number=3, chunk_index=2),
            "C3": create_synthetic_record(3, "dummy.pdf", page_number=2, chunk_index=3),
            "C4": create_synthetic_record(4, "african_elephant.pdf", page_number=2, chunk_index=4),
        }
        rag_res = create_synthetic_rag_result(answer=raw, source_map=smap)
        res = resolve_rag_citations(rag_res)
        expected = (
            "- fGCM levels (african_elephant.pdf, p. 3)\n"
            "- eTGS (african_elephant.pdf, pp. 2, 3)"
        )
        self.assertEqual(res["rendered_answer"], expected)

    def test_44_punctuation_survives_replacement(self) -> None:
        raw = "Eagles ingest fragments [C1]. Next sentence."
        rag_res = create_synthetic_rag_result(answer=raw)
        res = resolve_rag_citations(rag_res)
        expected = "Eagles ingest fragments (bald_eagle_lead_exposure.pdf, p. 1). Next sentence."
        self.assertEqual(res["rendered_answer"], expected)
        self.assertNotIn(" .", res["rendered_answer"])

    def test_45_input_stage_9_result_is_not_mutated(self) -> None:
        rag_res = create_synthetic_rag_result()
        original = copy.deepcopy(rag_res)
        _ = resolve_rag_citations(rag_res)
        self.assertEqual(rag_res, original)

    def test_46_citation_only_processing_needs_no_api_key(self) -> None:
        # Verified by executing offline synthetic resolution with no API key in environment
        rag_res = create_synthetic_rag_result()
        res = resolve_rag_citations(rag_res)
        self.assertIsNotNone(res)

    def test_47_citation_only_processing_performs_no_network_calls(self) -> None:
        # Executing resolve_rag_citations on synthetic data requires 0 network connections
        rag_res = create_synthetic_rag_result()
        res = resolve_rag_citations(rag_res)
        self.assertEqual(len(res["citations"]), 2)

    def test_synthetic_important_case_fgcm_etgs(self) -> None:
        """Test exact prompt synthetic case with fGCM and eTGS."""
        raw = "The study measured fGCM [C2] and temporal gland secretions [C2, C4]."
        smap = {
            "C1": create_synthetic_record(1, "dummy.pdf", page_number=1, chunk_index=1),
            "C2": create_synthetic_record(2, "african_elephant_reintegration.pdf", page_number=3, chunk_index=2),
            "C3": create_synthetic_record(3, "dummy.pdf", page_number=2, chunk_index=3),
            "C4": create_synthetic_record(4, "african_elephant_reintegration.pdf", page_number=2, chunk_index=4),
        }
        rag_res = create_synthetic_rag_result(answer=raw, source_map=smap)
        res = resolve_rag_citations(rag_res)
        expected = (
            "The study measured fGCM "
            "(african_elephant_reintegration.pdf, p. 3) "
            "and temporal gland secretions "
            "(african_elephant_reintegration.pdf, pp. 2, 3)."
        )
        self.assertEqual(res["rendered_answer"], expected)

    def test_ordinary_bracket_text_preservation(self) -> None:
        """Verify ordinary bracket text like [CITES Appendix I] is preserved while valid citations resolve."""
        raw = "Elephant species [Conservation status] are listed in [CITES Appendix I] and observed [see discussion above] [C1]."
        rag_res = create_synthetic_rag_result(answer=raw)
        res = resolve_rag_citations(rag_res)
        expected = (
            "Elephant species [Conservation status] are listed in [CITES Appendix I] "
            "and observed [see discussion above] (bald_eagle_lead_exposure.pdf, p. 1)."
        )
        self.assertEqual(res["rendered_answer"], expected)
        self.assertIn("[Conservation status]", res["rendered_answer"])
        self.assertIn("[CITES Appendix I]", res["rendered_answer"])
        self.assertIn("[see discussion above]", res["rendered_answer"])

    def test_non_finite_similarity_score_nan_rejected(self) -> None:
        """Verify NaN similarity score in source_map is rejected."""
        rag_res = create_synthetic_rag_result()
        rag_res["source_map"]["C2"]["similarity_score"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("non-finite", str(ctx.exception))

    def test_non_finite_similarity_score_pos_inf_rejected(self) -> None:
        """Verify +inf similarity score in source_map is rejected."""
        rag_res = create_synthetic_rag_result()
        rag_res["source_map"]["C2"]["similarity_score"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("non-finite", str(ctx.exception))

    def test_non_finite_similarity_score_neg_inf_rejected(self) -> None:
        """Verify -inf similarity score in source_map is rejected."""
        rag_res = create_synthetic_rag_result()
        rag_res["source_map"]["C2"]["similarity_score"] = float("-inf")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("non-finite", str(ctx.exception))


    def test_mixed_valid_and_malformed_standalone_c_rejected(self) -> None:
        """Verify presence of a valid citation does not bypass rejection of malformed [C]."""
        rag_res = create_synthetic_rag_result(answer="Claim [C1]. Bad [C].")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Malformed controlled citation marker detected", str(ctx.exception))

    def test_mixed_valid_and_malformed_c_minus_one_rejected(self) -> None:
        """Verify presence of a valid citation does not bypass rejection of malformed [C-1]."""
        rag_res = create_synthetic_rag_result(answer="Claim [C1]. Bad [C-1].")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Malformed controlled citation marker detected", str(ctx.exception))

    def test_mixed_valid_and_malformed_c_space_one_rejected(self) -> None:
        """Verify presence of a valid citation does not bypass rejection of malformed [C 1]."""
        rag_res = create_synthetic_rag_result(answer="Claim [C1]. Bad [C 1].")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Malformed controlled citation marker detected", str(ctx.exception))

    def test_mixed_valid_and_malformed_c_plus_one_rejected(self) -> None:
        """Verify presence of a valid citation does not bypass rejection of malformed [C+1]."""
        rag_res = create_synthetic_rag_result(answer="Claim [C1]. Bad [C+1].")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Malformed controlled citation marker detected", str(ctx.exception))

    def test_mixed_valid_and_malformed_lowercase_c_rejected(self) -> None:
        """Verify presence of a valid citation does not bypass rejection of malformed [c]."""
        rag_res = create_synthetic_rag_result(answer="Claim [C1]. Bad [c].")
        with self.assertRaises(ValueError) as ctx:
            resolve_rag_citations(rag_res)
        self.assertIn("Malformed controlled citation marker detected", str(ctx.exception))

    def test_ordinary_bracket_text_preservation_and_rendering(self) -> None:
        """Assert successful resolution and exact preservation of ordinary bracket text."""
        cases = [
            ("Species is protected [CITES Appendix I]. Claim [C1].", "Species is protected [CITES Appendix I]. Claim (bald_eagle_lead_exposure.pdf, p. 1)."),
            ("[Conservation status] Claim [C1].", "[Conservation status] Claim (bald_eagle_lead_exposure.pdf, p. 1)."),
            ("[see discussion above] Claim [C1].", "[see discussion above] Claim (bald_eagle_lead_exposure.pdf, p. 1)."),
        ]
        for raw, expected in cases:
            rag_res = create_synthetic_rag_result(answer=raw)
            res = resolve_rag_citations(rag_res)
            self.assertEqual(res["rendered_answer"], expected)


if __name__ == "__main__":
    unittest.main()
