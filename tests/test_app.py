"""Stage 11 — Automated test suite for Streamlit web application.

Uses native Streamlit AppTest and dependency injection to test all UI contracts:
- Initial load, scope, and forms
- Input validation (empty vs accepted)
- Fixed top_k=4 backend call counts
- Single submission vs harmless rerun behavior
- Rendered answer vs raw answer / marker boundary
- Supported answer & Sources display
- Exact fallback display without Sources
- Grouped source display and URL validation
- Safe user-facing error mapping without secret/traceback leakage
- Offline operation without live API key or FAISS index
"""

import inspect
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from app import (
    group_citation_records,
    is_valid_web_url,
    map_backend_error_to_user_message,
    render_app,
)
from src.generation import INSUFFICIENT_CONTEXT_FALLBACK
from src.offline_fallback import answer_with_free_fallback


# Synthetic fixture helpers matching Stage 10 data contracts
def create_synthetic_supported_result(
    question: str = "How does lead ammunition expose bald eagles to lead?",
    answer: str = "Bald eagles scavenge gut piles containing lead ammunition fragments [C1, C2].",
    rendered_answer: str = "Bald eagles scavenge gut piles containing lead ammunition fragments (bald_eagle_lead_exposure.pdf, p. 1).",
    answer_mode: str = "gemini",
) -> Dict[str, Any]:
    return {
        "question": question,
        "raw_answer": answer,
        "rendered_answer": rendered_answer,
        "generation_model": "gemini-3.8-flash",
        "citation_ids": ["C1", "C2"],
        "citations": [
            {
                "source_id": "C1",
                "chunk_id": "doc_eagle_p001_c001",
                "document_id": "doc_eagle",
                "filename": "bald_eagle_lead_exposure.pdf",
                "page_number": 1,
                "title": "Synthetic Bald Eagle Lead Study",
                "publisher": "USGS",
                "source_url": "https://example.org/eagle.pdf",
                "rank": 1,
                "chunk_index": 1,
            },
            {
                "source_id": "C2",
                "chunk_id": "doc_eagle_p001_c002",
                "document_id": "doc_eagle",
                "filename": "bald_eagle_lead_exposure.pdf",
                "page_number": 1,
                "title": "Synthetic Bald Eagle Lead Study",
                "publisher": "USGS",
                "source_url": "https://example.org/eagle.pdf",
                "rank": 2,
                "chunk_index": 2,
            },
        ],
        "citation_group_count": 1,
        "unique_citation_count": 2,
        "is_fallback": False,
        "answer_mode": answer_mode,
    }


def create_synthetic_fallback_result(
    question: str = "What do snow leopards eat in winter?",
    answer_mode: str = "gemini",
) -> Dict[str, Any]:
    return {
        "question": question,
        "raw_answer": INSUFFICIENT_CONTEXT_FALLBACK,
        "rendered_answer": INSUFFICIENT_CONTEXT_FALLBACK,
        "generation_model": "gemini-3.8-flash",
        "citation_ids": [],
        "citations": [],
        "citation_group_count": 0,
        "unique_citation_count": 0,
        "is_fallback": True,
        "answer_mode": answer_mode,
    }


def _run_test_app(answer_fn=None):
    import app
    if answer_fn is not None:
        app.render_app(answer_fn=answer_fn)
    else:
        app.render_app()


class TestAppStage11(unittest.TestCase):
    """Test suite covering Stage 11 Streamlit Web Application requirements."""

    def test_01_url_validation_accepts_valid_http_https(self) -> None:
        self.assertTrue(is_valid_web_url("https://example.org/study.pdf"))
        self.assertTrue(is_valid_web_url("http://example.org/study.pdf"))

    def test_02_url_validation_rejects_invalid_schemes_and_malformed_urls(self) -> None:
        self.assertFalse(is_valid_web_url("ftp://example.org/study.pdf"))
        self.assertFalse(is_valid_web_url("file:///etc/passwd"))
        self.assertFalse(is_valid_web_url("javascript:alert(1)"))
        self.assertFalse(is_valid_web_url("not_a_url"))
        self.assertFalse(is_valid_web_url(""))
        self.assertFalse(is_valid_web_url(None))

    def test_03_citation_grouping_collapses_duplicates_and_preserves_order(self) -> None:
        cits = [
            {
                "source_id": "C1",
                "chunk_id": "chunk_1",
                "filename": "file_a.pdf",
                "page_number": 1,
                "title": "Title A",
                "publisher": "Pub A",
                "source_url": "https://a.org",
            },
            {
                "source_id": "C2",
                "chunk_id": "chunk_2",
                "filename": "file_a.pdf",
                "page_number": 1,
                "title": "Title A",
                "publisher": "Pub A",
                "source_url": "https://a.org",
            },
            {
                "source_id": "C3",
                "chunk_id": "chunk_3",
                "filename": "file_b.pdf",
                "page_number": 2,
                "title": "Title B",
                "publisher": "Pub B",
                "source_url": "https://b.org",
            },
        ]
        grouped = group_citation_records(cits)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["filename"], "file_a.pdf")
        self.assertEqual(grouped[0]["source_ids"], ["C1", "C2"])
        self.assertEqual(grouped[0]["chunk_ids"], ["chunk_1", "chunk_2"])
        self.assertEqual(grouped[1]["filename"], "file_b.pdf")
        self.assertEqual(grouped[1]["source_ids"], ["C3"])

    def test_04_citation_grouping_does_not_mutate_input(self) -> None:
        cits = [
            {
                "source_id": "C1",
                "chunk_id": "chunk_1",
                "filename": "file_a.pdf",
                "page_number": 1,
                "title": "Title A",
                "publisher": "Pub A",
                "source_url": "https://a.org",
            }
        ]
        original_cits = [dict(cits[0])]
        _ = group_citation_records(cits)
        self.assertEqual(cits, original_cits)

    def test_05_error_mapping_hides_secrets_and_paths(self) -> None:
        exc_secret = ValueError("GEMINI_API_KEY=AIzaSySecretKey is invalid at /home/user/path")
        mapped = map_backend_error_to_user_message(exc_secret)
        self.assertEqual(mapped, "Application is not configured with a valid Gemini API key.")
        self.assertNotIn("AIzaSySecretKey", mapped)
        self.assertNotIn("/home/user/path", mapped)

    def test_06_error_mapping_handles_index_and_citation_failures(self) -> None:
        exc_index = FileNotFoundError("index/faiss.index missing")
        self.assertEqual(map_backend_error_to_user_message(exc_index), "Animal knowledge index is unavailable.")

        exc_cit = ValueError("Malformed controlled citation marker detected: '[C]'")
        self.assertEqual(
            map_backend_error_to_user_message(exc_cit),
            "Generated answer could not be safely validated against its sources and was withheld.",
        )

        exc_gen = RuntimeError("API service 503 unavailable")
        self.assertEqual(
            map_backend_error_to_user_message(exc_gen),
            "Answer could not be generated right now; please try again.",
        )

    def test_07_apptest_initial_load_renders_title_scope_form_no_backend_call(self) -> None:
        mock_backend = MagicMock()
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        self.assertFalse(at.exception)
        mock_backend.assert_not_called()
        self.assertTrue(len(at.title) > 0)
        self.assertIn("Animal Knowledge RAG Assistant", at.title[0].value)
        combined_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn("Deterministic offline lexical/extractive quota fallback", combined_markdown)

    def test_08_apptest_empty_query_shows_warning_no_backend_call(self) -> None:
        mock_backend = MagicMock()
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("   ")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        mock_backend.assert_not_called()
        self.assertTrue(len(at.warning) > 0)
        self.assertIn("Please enter a question.", at.warning[0].value)

    def test_09_apptest_valid_submission_executes_backend_with_top_k_4(self) -> None:
        mock_backend = MagicMock(return_value=create_synthetic_supported_result())
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        q = "How does lead ammunition expose bald eagles to lead?"
        at.text_area[0].input(q)
        at.button[0].click().run()

        self.assertFalse(at.exception)
        mock_backend.assert_called_once_with(
            question=q,
            top_k=4,
            index_dir="index",
        )

    def test_10_apptest_supported_result_renders_answer_and_sources(self) -> None:
        synth_res = create_synthetic_supported_result(
            answer="Bald eagles scavenge gut piles containing lead [C1].",
            rendered_answer="Bald eagles scavenge gut piles containing lead (bald_eagle_lead_exposure.pdf, p. 1).",
        )
        mock_backend = MagicMock(return_value=synth_res)
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("How does lead ammunition expose bald eagles to lead?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        markdown_texts = [m.value for m in at.markdown]
        combined_markdown = "\n".join(markdown_texts)

        self.assertIn("Bald eagles scavenge gut piles containing lead (bald_eagle_lead_exposure.pdf, p. 1).", combined_markdown)
        self.assertNotIn("[C1]", combined_markdown)
        self.assertIn("Sources & Grounding", combined_markdown)
        self.assertIn("Synthetic Bald Eagle Lead Study", combined_markdown)

    def test_11_apptest_fallback_result_renders_exact_fallback_no_sources(self) -> None:
        synth_fallback = create_synthetic_fallback_result("What do snow leopards eat?")
        mock_backend = MagicMock(return_value=synth_fallback)
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("What do snow leopards eat?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        markdown_texts = [m.value for m in at.markdown]
        combined_markdown = "\n".join(markdown_texts)

        self.assertIn(INSUFFICIENT_CONTEXT_FALLBACK, combined_markdown)
        self.assertNotIn("Sources & Grounding", combined_markdown)

    def test_12_apptest_backend_error_displays_safe_message_clears_stale_answer(self) -> None:
        mock_backend = MagicMock()
        mock_backend.side_effect = [
            create_synthetic_supported_result(),
            ValueError("Malformed controlled citation marker detected: '[C]'"),
        ]

        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        # Query 1
        at.text_area[0].input("First valid question?")
        at.button[0].click().run()
        self.assertEqual(len(at.error), 0)

        # Query 2 fails
        at.text_area[0].input("Second question that fails?")
        at.button[0].click().run()

        self.assertTrue(len(at.error) > 0)
        self.assertIn(
            "Generated answer could not be safely validated against its sources and was withheld.",
            at.error[0].value,
        )
        markdown_texts = [m.value for m in at.markdown]
        combined_markdown = "\n".join(markdown_texts)
        self.assertNotIn("Bald eagles scavenge gut piles", combined_markdown)

    def test_13_apptest_exception_with_secret_sentinel_never_leaks_raw_text(self) -> None:
        mock_backend = MagicMock()
        mock_backend.side_effect = Exception("SECRET_SENTINEL_KEY_12345 leaked at /path/to/private/file")

        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("Question triggering secret exception?")
        at.button[0].click().run()

        self.assertTrue(len(at.error) > 0)
        err_val = at.error[0].value
        self.assertNotIn("SECRET_SENTINEL_KEY_12345", err_val)
        self.assertNotIn("/path/to/private/file", err_val)
        self.assertEqual(err_val, "Answer could not be generated right now; please try again.")

    def test_14_apptest_submission_preserves_exact_untrimmed_question(self) -> None:
        raw_question = "  How does lead affect bald eagles?  "
        mock_backend = MagicMock(return_value=create_synthetic_supported_result(question=raw_question))

        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input(raw_question)
        at.button[0].click().run()

        self.assertFalse(at.exception)
        mock_backend.assert_called_once_with(
            question=raw_question,
            top_k=4,
            index_dir="index",
        )
        # Verify kwargs passed to mock_backend
        _, kwargs = mock_backend.call_args
        self.assertEqual(kwargs["question"], raw_question)
        self.assertEqual(kwargs["top_k"], 4)
        self.assertEqual(at.session_state["submitted_question"], raw_question)

    def test_15_apptest_gemini_result_does_not_show_offline_notice(self) -> None:
        mock_backend = MagicMock(return_value=create_synthetic_supported_result(answer_mode="gemini"))
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("How does lead affect bald eagles?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 0)

    def test_16_apptest_offline_supported_result_shows_notice_answer_and_sources(self) -> None:
        synth_res = create_synthetic_supported_result(answer_mode="offline_extractive")
        mock_backend = MagicMock(return_value=synth_res)
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("How does lead affect bald eagles?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 1)
        self.assertEqual(
            at.info[0].value,
            "Gemini is temporarily unavailable due to an API limit, so this answer was extracted directly from the indexed sources.",
        )
        combined_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn(synth_res["rendered_answer"], combined_markdown)
        self.assertIn("Sources & Grounding", combined_markdown)
        self.assertIn("Synthetic Bald Eagle Lead Study", combined_markdown)

    def test_17_apptest_offline_insufficient_context_shows_notice_without_sources(self) -> None:
        synth_res = create_synthetic_fallback_result(answer_mode="offline_extractive")
        mock_backend = MagicMock(return_value=synth_res)
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("What do snow leopards eat?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 1)
        combined_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn(INSUFFICIENT_CONTEXT_FALLBACK, combined_markdown)
        self.assertNotIn("Sources & Grounding", combined_markdown)

    def test_18_apptest_result_without_answer_mode_remains_supported(self) -> None:
        synth_res = create_synthetic_supported_result()
        synth_res.pop("answer_mode")
        mock_backend = MagicMock(return_value=synth_res)
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()
        at.text_area[0].input("How does lead affect bald eagles?")
        at.button[0].click().run()

        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 0)
        combined_markdown = "\n".join(m.value for m in at.markdown)
        self.assertIn(synth_res["rendered_answer"], combined_markdown)
        self.assertIn("Sources & Grounding", combined_markdown)

    def test_19_render_app_defaults_to_free_fallback_backend(self) -> None:
        default_backend = inspect.signature(render_app).parameters["answer_fn"].default
        self.assertIs(default_backend, answer_with_free_fallback)

    def test_20_apptest_gemini_submission_clears_previous_offline_notice(self) -> None:
        mock_backend = MagicMock(
            side_effect=[
                create_synthetic_supported_result(answer_mode="offline_extractive"),
                create_synthetic_supported_result(answer_mode="gemini"),
            ]
        )
        at = AppTest.from_function(_run_test_app, args=(mock_backend,), default_timeout=5).run()

        at.text_area[0].input("First offline question?")
        at.button[0].click().run()
        self.assertEqual(len(at.info), 1)

        at.text_area[0].input("Second Gemini question?")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.info), 0)
        self.assertEqual(mock_backend.call_count, 2)


if __name__ == "__main__":
    unittest.main()
