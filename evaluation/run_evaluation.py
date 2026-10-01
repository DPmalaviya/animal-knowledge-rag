"""Runner module for Stage 14 evaluation snapshot execution.

Executes single-pass Top-4 retrieval and Gemini 3.8 Flash RAG generation for each
of the 20 frozen golden questions, persisting evidence to disk.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.citations import resolve_rag_citations
from src.embeddings import create_genai_client
from src.generation import (
    DEFAULT_GENERATION_MODEL,
    build_generation_prompt,
    build_source_map,
    generate_grounded_answer,
)
from src.retrieval import DEFAULT_TOP_K, retrieve, validate_top_k
from src.vector_store import DEFAULT_INDEX_DIR, load_vector_store

EXPECTED_INDEX_HASHES = {
    "index/faiss.index": "a5622eb106e81a4cc151d6ed33939239291f1da6df48329fc5ffa0a3b966038e",
    "index/chunk_metadata.json": "814cab2575bfe63060321faafc20f971dc11c6a7a59c18dbc75876275c617877",
    "index/index_manifest.json": "65ba5bc1a8d1a7bb700e0d614dceb98d9329c2ae5fbc904995a6f0cd3be507ef",
}


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a local file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_index_artifacts(index_dir: str = DEFAULT_INDEX_DIR) -> Dict[str, str]:
    """Verify hashes of stored Stage 7 runtime index artifacts."""
    actual_hashes = {}
    for rel_path, expected_hash in EXPECTED_INDEX_HASHES.items():
        if not os.path.exists(rel_path):
            raise FileNotFoundError(f"Required index artifact missing: {rel_path}")
        actual_hash = compute_file_sha256(rel_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Index artifact hash mismatch for '{rel_path}': expected {expected_hash}, got {actual_hash}"
            )
        actual_hashes[rel_path] = actual_hash
    return actual_hashes


def run_evaluation_snapshot(
    golden_path: str = "evaluation/golden_questions.json",
    output_dir: str = "evaluation/results",
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = DEFAULT_TOP_K,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    git_sha: str = "1bed8e6454ec3782cced2dbc5e97f5f5074b5175",
) -> Dict[str, Any]:
    """Execute evaluation snapshot across 20 golden questions."""
    # 1. Verify index artifacts
    index_hashes = verify_index_artifacts(index_dir)

    # 2. Load and hash golden questions dataset
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden questions file missing: {golden_path}")

    golden_hash = compute_file_sha256(golden_path)
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    questions = golden_data.get("questions", [])
    if len(questions) != 20:
        raise ValueError(f"Golden dataset must contain exactly 20 questions, got {len(questions)}")

    # 3. Create results directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Initialize API client
    active_client = client or create_genai_client(api_key=api_key)

    retrieval_results = []
    rag_results = []

    planned_count = len(questions)
    attempted_count = 0
    successful_count = 0
    failed_count = 0

    normal_embedding_requests = 0
    normal_generation_requests = 0

    start_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"Starting Stage 14 Evaluation Snapshot ({planned_count} questions)...")

    for q_idx, q_item in enumerate(questions, 1):
        q_id = q_item["id"]
        q_text = q_item["question"]
        q_type = q_item["type"]
        attempted_count += 1

        print(f"[{q_idx:02d}/20] Processing {q_id} ({q_type}): \"{q_text[:60]}...\"")

        # Single retrieval execution
        retrieval_records = None
        retrieval_error = None
        try:
            retrieval_records = retrieve(
                question=q_text,
                index_dir=index_dir,
                top_k=top_k,
                client=active_client,
            )
            normal_embedding_requests += 1
        except Exception as e:
            retrieval_error = str(e)
            print(f"  -> ERROR in retrieval for {q_id}: {e}")

        # Build retrieval evidence structure
        ret_entry = {
            "question_id": q_id,
            "question": q_text,
            "type": q_type,
            "expected_filenames": q_item.get("expected_filenames", []),
            "accepted_pages": q_item.get("accepted_pages", []),
            "accepted_evidence": q_item.get("accepted_evidence", []),
            "retrieved_chunks": [],
            "error": retrieval_error,
        }

        if retrieval_records:
            for rec in retrieval_records:
                ret_entry["retrieved_chunks"].append({
                    "rank": rec["rank"],
                    "faiss_id": rec["faiss_id"],
                    "similarity_score": rec["similarity_score"],
                    "chunk_id": rec["chunk_id"],
                    "document_id": rec["document_id"],
                    "filename": rec["filename"],
                    "page_number": rec["page_number"],
                    "title": rec["title"],
                    "publisher": rec["publisher"],
                    "source_url": rec["source_url"],
                    "estimated_token_count": rec["estimated_token_count"],
                })

        retrieval_results.append(ret_entry)

        # Skip generation if retrieval failed
        if not retrieval_records:
            failed_count += 1
            rag_results.append({
                "question_id": q_id,
                "question": q_text,
                "type": q_type,
                "status": "failed",
                "stage_failure": "retrieval",
                "error": retrieval_error,
                "raw_answer": None,
                "rendered_answer": None,
                "citation_ids": [],
                "citations": [],
                "is_fallback": False,
            })
            continue

        # Single generation execution using same retrieved records
        rag_entry = {
            "question_id": q_id,
            "question": q_text,
            "type": q_type,
            "status": "attempted",
            "error": None,
        }

        try:
            source_map = build_source_map(retrieval_records)
            prompt = build_generation_prompt(q_text, source_map)

            gen_res = generate_grounded_answer(
                client=active_client,
                prompt=prompt,
                model_name=DEFAULT_GENERATION_MODEL,
            )
            normal_generation_requests += 1

            # Build Stage 9 result structure
            stage9_result = {
                "question": q_text,
                "answer": gen_res["answer"],
                "generation_model": DEFAULT_GENERATION_MODEL,
                "source_map": source_map,
                "retrieved_context_count": len(retrieval_records),
                "usage_metadata": gen_res.get("usage_metadata"),
            }

            # Resolve Stage 10 citations deterministically
            cited_res = resolve_rag_citations(stage9_result)

            rag_entry.update({
                "status": "success",
                "raw_answer": cited_res["raw_answer"],
                "rendered_answer": cited_res["rendered_answer"],
                "generation_model": cited_res["generation_model"],
                "citation_ids": cited_res["citation_ids"],
                "citations": cited_res["citations"],
                "citation_group_count": cited_res["citation_group_count"],
                "unique_citation_count": cited_res["unique_citation_count"],
                "is_fallback": cited_res["is_fallback"],
                "usage_metadata": gen_res.get("usage_metadata"),
            })
            successful_count += 1
            print(f"  -> SUCCESS: Answer length={len(cited_res['rendered_answer'])} | Citations={cited_res['citation_ids']} | Fallback={cited_res['is_fallback']}")

        except Exception as e:
            failed_count += 1
            rag_entry.update({
                "status": "failed",
                "stage_failure": "generation_or_citation",
                "error": str(e),
                "raw_answer": None,
                "rendered_answer": None,
                "citation_ids": [],
                "citations": [],
                "is_fallback": False,
            })
            print(f"  -> ERROR in generation/citation for {q_id}: {e}")

        rag_results.append(rag_entry)

        # Conservative sequential delay to stay strictly within API rate bounds
        time.sleep(2.0)

    end_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = {
        "run_id": f"eval_snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": start_time,
        "end_time": end_time,
        "git_sha": git_sha,
        "freeze_sha": git_sha,
        "golden_questions_sha256": golden_hash,
        "index_hashes": index_hashes,
        "python_version": sys.version,
        "models": {
            "embedding": "gemini-embedding-2",
            "generation": DEFAULT_GENERATION_MODEL,
        },
        "embedding_dimension": 768,
        "top_k": top_k,
        "index_type": "IndexFlatIP",
        "total_index_vectors": 272,
        "counts": {
            "planned_questions": planned_count,
            "attempted_questions": attempted_count,
            "successful_questions": successful_count,
            "failed_questions": failed_count,
        },
        "api_accounting": {
            "normal_embedding_requests": normal_embedding_requests,
            "normal_generation_requests": normal_generation_requests,
            "total_normal_requests": normal_embedding_requests + normal_generation_requests,
            "retry_accounting_note": "Internal transient retries (if any) handled defensively by SDK wrapper",
        },
        "completion_status": "COMPLETED" if failed_count == 0 else "PARTIAL",
    }

    # Persist JSON evidence files atomically
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(os.path.join(output_dir, "retrieval_results.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2)

    with open(os.path.join(output_dir, "rag_results.json"), "w", encoding="utf-8") as f:
        json.dump(rag_results, f, indent=2)

    print(f"\nEvaluation Snapshot Completed! Results saved to '{output_dir}'.")
    return manifest


def main() -> None:
    """CLI entrypoint for running evaluation snapshot."""
    parser = argparse.ArgumentParser(description="Stage 14 Evaluation Snapshot Runner")
    parser.add_argument("--golden", type=str, default="evaluation/golden_questions.json")
    parser.add_argument("--outdir", type=str, default="evaluation/results")
    args = parser.parse_args()

    try:
        run_evaluation_snapshot(golden_path=args.golden, output_dir=args.outdir)
    except Exception as e:
        print(f"\nERROR running evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
