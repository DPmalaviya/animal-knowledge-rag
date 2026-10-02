"""Stage 11 — Streamlit Web Application for Animal Knowledge RAG Assistant.

Provides a clean, native Streamlit interface for grounded animal knowledge QA
with deterministic source citations, safe URL validation, source grouping,
and safe fail-closed error handling.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import streamlit as st

from src.generation import INSUFFICIENT_CONTEXT_FALLBACK
from src.offline_fallback import answer_with_free_fallback

# Default fixed index directory and Top-K context count
DEFAULT_INDEX_DIR = "index"
FIXED_TOP_K = 4

# Target animal species in curated demo corpus
CURATED_SPECIES = [
    "Bald Eagles",
    "Monarch Butterflies",
    "Humpback Whales",
    "Green Sea Turtles",
    "African Elephants",
]

EXAMPLE_QUESTIONS = [
    "How does lead ammunition expose bald eagles to lead?",
    "What factors impact monarch butterfly flight performance?",
    "How are fecal glucocorticoid metabolites measured in African elephants?",
]


def is_valid_web_url(url: Optional[str]) -> bool:
    """Validate web URL string for safe clickable link display.

    Accepts HTTP/HTTPS schemes with valid non-empty netloc.
    Rejects malformed URLs, local file paths, javascript, or non-web schemes.
    """
    if not url or not isinstance(url, str):
        return False
    url_str = url.strip()
    if not url_str:
        return False
    try:
        parsed = urlparse(url_str)
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        if not parsed.netloc or not parsed.netloc.strip():
            return False
        # Reject control characters or credentials embedded in authority if suspicious
        if "\0" in url_str or "\n" in url_str or "\r" in url_str:
            return False
        return True
    except Exception:
        return False


def group_citation_records(
    citations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Group identical source records by (filename, page_number, title, publisher, source_url).

    Preserves first-appearance order, collects all associated source_ids and chunk_ids,
    and returns a list of grouped citation display dictionaries without mutating inputs.
    """
    grouped_list: List[Dict[str, Any]] = []
    seen_map: Dict[Tuple[str, int, str, str, str], Dict[str, Any]] = {}

    for cit in citations:
        key = (
            str(cit.get("filename", "")),
            int(cit.get("page_number", 0)),
            str(cit.get("title", "")),
            str(cit.get("publisher", "")),
            str(cit.get("source_url", "")),
        )
        if key not in seen_map:
            group_entry = {
                "filename": cit.get("filename", ""),
                "page_number": cit.get("page_number", 0),
                "title": cit.get("title", "Untitled Document"),
                "publisher": cit.get("publisher", "Unknown Publisher"),
                "source_url": cit.get("source_url", ""),
                "source_ids": [cit.get("source_id", "")],
                "chunk_ids": [cit.get("chunk_id", "")],
            }
            seen_map[key] = group_entry
            grouped_list.append(group_entry)
        else:
            existing = seen_map[key]
            sid = cit.get("source_id", "")
            cid = cit.get("chunk_id", "")
            if sid and sid not in existing["source_ids"]:
                existing["source_ids"].append(sid)
            if cid and cid not in existing["chunk_ids"]:
                existing["chunk_ids"].append(cid)

    return grouped_list


def map_backend_error_to_user_message(exc: Exception) -> str:
    """Map backend execution exceptions to user-safe error messages.

    Ensures no raw tracebacks, internal paths, or secrets are exposed to the UI.
    """
    msg = str(exc).lower()

    # API Key missing / unconfigured / unauthorized
    if "api_key" in msg or "gemini_api_key" in msg or "api key" in msg or "401" in msg or "403" in msg or "unauthorized" in msg:
        return "Application is not configured with a valid Gemini API key."

    # Rate limit / quota / 429
    if "429" in msg or "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg:
        return "Gemini API rate limit or quota exceeded. Please wait a moment and try again."

    # Index unavailable / corrupt
    if isinstance(exc, FileNotFoundError) or "index" in msg or "faiss" in msg or "metadata" in msg:
        return "Animal knowledge index is unavailable."

    # Citation validation failures
    if isinstance(exc, ValueError) and (
        "citation" in msg or "malformed" in msg or "fallback" in msg or "source_map" in msg or "unknown" in msg
    ):
        return "Generated answer could not be safely validated against its sources and was withheld."

    # General / provider error fallback
    return "Answer could not be generated right now; please try again."


def render_app(answer_fn: Callable[..., Dict[str, Any]] = answer_with_free_fallback) -> None:
    """Render the main Streamlit application interface.

    Args:
        answer_fn: Injectable backend function for RAG execution (defaults to the free fallback backend).
    """
    # Check Streamlit Cloud st.secrets for GEMINI_API_KEY if process env is unset
    api_key = None
    if "GEMINI_API_KEY" in os.environ and os.environ["GEMINI_API_KEY"].strip():
        api_key = os.environ["GEMINI_API_KEY"].strip()
    else:
        try:
            if "GEMINI_API_KEY" in st.secrets:
                sec_val = st.secrets["GEMINI_API_KEY"]
                if isinstance(sec_val, str) and sec_val.strip():
                    api_key = sec_val.strip()
                    os.environ["GEMINI_API_KEY"] = api_key
        except Exception:
            pass
    st.set_page_config(
        page_title="Animal Knowledge RAG Assistant",
        page_icon="🐾",
        layout="centered",
    )

    st.title("🐾 Animal Knowledge RAG Assistant")
    st.markdown(
        "Ask grounded questions about animal biology, conservation, and ecology. "
        "Answers are generated strictly from our curated 9-document demo corpus with physical page citations."
    )

    # Sidebar disclosures and scope
    with st.sidebar:
        st.header("About & Scope")
        st.markdown(
            "This RAG system operates over a curated dataset of **9 peer-reviewed & public research PDFs** "
            "(85 physical pages)."
        )
        st.markdown("**Covered Species:**")
        for sp in CURATED_SPECIES:
            st.markdown(f"- {sp}")
        st.divider()
        st.markdown(
            "**Architecture:**\n"
            "- FAISS IndexFlatIP (768-dim)\n"
            "- Gemini 3.8 Flash (Thinking Level: Low)\n"
            "- Fixed Top-K Retrieval: 4 chunks\n"
            "- Deterministic Stage 10 Citation Parsing"
        )

    # Example question expander
    with st.expander("💡 View Example Questions"):
        for eq in EXAMPLE_QUESTIONS:
            st.markdown(f"- *{eq}*")

    # Initialize or load session state for latest result
    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None
    if "submitted_question" not in st.session_state:
        st.session_state["submitted_question"] = None
    if "error_message" not in st.session_state:
        st.session_state["error_message"] = None

    # Submission form
    with st.form(key="rag_query_form"):
        user_query = st.text_area(
            label="Enter your question:",
            placeholder="e.g., How does lead ammunition expose bald eagles to lead?",
            height=100,
        )
        submit_button = st.form_submit_button(label="Ask Assistant", type="primary")

    if submit_button:
        is_empty = not user_query or not user_query.strip()
        if is_empty:
            st.warning("Please enter a question.")
            st.session_state["latest_result"] = None
            st.session_state["submitted_question"] = None
            st.session_state["error_message"] = None
        else:
            # Clear previous result state on new submission
            st.session_state["latest_result"] = None
            st.session_state["submitted_question"] = user_query
            st.session_state["error_message"] = None

            with st.spinner("Retrieving context and generating grounded answer..."):
                try:
                    res = answer_fn(
                        question=user_query,
                        top_k=FIXED_TOP_K,
                        index_dir=DEFAULT_INDEX_DIR,
                    )
                    # Validate expected presentation fields
                    if (
                        isinstance(res, dict)
                        and "rendered_answer" in res
                        and "is_fallback" in res
                        and "citations" in res
                    ):
                        st.session_state["latest_result"] = res
                    else:
                        st.session_state["error_message"] = (
                            "Answer could not be generated right now; please try again."
                        )
                except Exception as exc:
                    st.session_state["error_message"] = map_backend_error_to_user_message(exc)

    # Display error message if present
    if st.session_state["error_message"]:
        st.error(st.session_state["error_message"])

    # Display latest trusted result if available
    latest_res = st.session_state["latest_result"]
    if latest_res and not st.session_state["error_message"]:
        st.divider()
        st.subheader("Answer")

        if latest_res.get("answer_mode") == "offline_extractive":
            st.info(
                "Gemini's free limit is currently reached, so this answer was extracted "
                "directly from the indexed sources."
            )

        # Trusted answer rendering boundary: render rendered_answer ONLY with unsafe_html=False
        rendered_answer = latest_res.get("rendered_answer", "")
        st.markdown(rendered_answer)

        is_fallback = latest_res.get("is_fallback", False)
        citations = latest_res.get("citations", [])

        # Render Sources section ONLY for non-fallback supported answers with citations
        if not is_fallback and citations:
            st.markdown("### Sources & Grounding")
            grouped_sources = group_citation_records(citations)

            for group in grouped_sources:
                title = group["title"]
                filename = group["filename"]
                page = group["page_number"]
                publisher = group["publisher"]
                url = group["source_url"]

                with st.container(border=True):
                    # Title & Page
                    st.markdown(f"**{title}**")
                    st.caption(f"📄 File: `{filename}` | Page: **p. {page}** | Publisher: {publisher}")

                    # Safe URL rendering using native st.link_button or markdown link
                    if is_valid_web_url(url):
                        st.link_button("🌐 View Source Document", url=url)


if __name__ == "__main__":
    render_app()
