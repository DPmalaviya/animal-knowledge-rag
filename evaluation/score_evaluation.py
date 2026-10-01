"""Scorer module for Stage 14 evaluation metrics.

Calculates deterministic retrieval metrics, generation fallback metrics, citation acceptance rates,
and human evaluation scores from persisted evaluation results.
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def score_retrieval(
    retrieval_results: List[Dict[str, Any]],
    golden_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate deterministic retrieval metrics for 16 supported questions."""
    supported_golds = {q["id"]: q for q in golden_questions if q["type"] == "supported"}
    total_supported = len(supported_golds)

    if total_supported == 0:
        raise ValueError("No supported questions found in golden dataset.")

    doc_hit_count = 0
    page_hit_count = 0
    mrr_sum = 0.0

    retrieval_failures = []

    for r_res in retrieval_results:
        q_id = r_res["question_id"]
        if q_id not in supported_golds:
            continue

        gold = supported_golds[q_id]
        exp_files = set(gold.get("expected_filenames", []))
        accepted_ev = set((item["filename"], item["page"]) for item in gold.get("accepted_evidence", []))

        ret_chunks = r_res.get("retrieved_chunks", [])

        doc_hit = False
        page_hit = False
        first_doc_rank = None

        for chunk in ret_chunks:
            rank = chunk["rank"]
            fn = chunk["filename"]
            page = chunk["page_number"]

            if fn in exp_files:
                doc_hit = True
                if first_doc_rank is None:
                    first_doc_rank = rank

            if (fn, page) in accepted_ev:
                page_hit = True

        if doc_hit:
            doc_hit_count += 1
            mrr_sum += 1.0 / first_doc_rank
        else:
            retrieval_failures.append({
                "question_id": q_id,
                "type": "doc_miss",
                "expected": list(exp_files),
                "got": [c["filename"] for c in ret_chunks],
            })

        if page_hit:
            page_hit_count += 1
        elif doc_hit:
            retrieval_failures.append({
                "question_id": q_id,
                "type": "page_miss",
                "expected_evidence": list(accepted_ev),
                "got_evidence": [(c["filename"], c["page_number"]) for c in ret_chunks],
            })

    return {
        "total_supported_questions": total_supported,
        "document_hit_at_4_count": doc_hit_count,
        "document_hit_at_4_rate": doc_hit_count / total_supported,
        "document_mrr_at_4": mrr_sum / total_supported,
        "page_hit_at_4_count": page_hit_count,
        "page_hit_at_4_rate": page_hit_count / total_supported,
        "retrieval_failures": retrieval_failures,
    }


def score_rag_generation(
    rag_results: List[Dict[str, Any]],
    golden_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate deterministic RAG generation metrics."""
    supported_golds = {q["id"]: q for q in golden_questions if q["type"] == "supported"}
    unsupported_golds = {q["id"]: q for q in golden_questions if q["type"] == "unsupported"}

    total_supported = len(supported_golds)
    total_unsupported = len(unsupported_golds)

    supported_non_fallback_count = 0
    supported_fallback_count = 0
    generation_error_count = 0

    exact_fallback_count = 0
    clean_fallback_count = 0

    citation_valid_count = 0
    total_attempted_supported_generations = 0

    generation_failures = []

    for r_res in rag_results:
        q_id = r_res["question_id"]
        q_type = r_res.get("type")

        if r_res.get("status") != "success":
            generation_error_count += 1
            generation_failures.append({
                "question_id": q_id,
                "type": "generation_error",
                "error": r_res.get("error"),
            })
            continue

        is_fallback = r_res.get("is_fallback", False)
        citation_ids = r_res.get("citation_ids", [])
        citations = r_res.get("citations", [])

        if q_type == "supported":
            if not is_fallback:
                supported_non_fallback_count += 1
                total_attempted_supported_generations += 1
                if citation_ids and len(citation_ids) == len(citations):
                    citation_valid_count += 1
            else:
                supported_fallback_count += 1
                generation_failures.append({
                    "question_id": q_id,
                    "type": "supported_unexpected_fallback",
                    "raw_answer": r_res.get("raw_answer"),
                })

        elif q_type == "unsupported":
            if is_fallback:
                exact_fallback_count += 1
                if len(citation_ids) == 0 and len(citations) == 0:
                    clean_fallback_count += 1
            else:
                generation_failures.append({
                    "question_id": q_id,
                    "type": "unsupported_failed_fallback",
                    "rendered_answer": r_res.get("rendered_answer"),
                })

    citation_acceptance_rate = (
        citation_valid_count / total_attempted_supported_generations
        if total_attempted_supported_generations > 0
        else 0.0
    )

    return {
        "supported": {
            "total": total_supported,
            "non_fallback_count": supported_non_fallback_count,
            "non_fallback_rate": supported_non_fallback_count / total_supported if total_supported > 0 else 0.0,
            "fallback_count": supported_fallback_count,
            "citation_acceptance_numerator": citation_valid_count,
            "citation_acceptance_denominator": total_attempted_supported_generations,
            "citation_acceptance_rate": citation_acceptance_rate,
        },
        "unsupported": {
            "total": total_unsupported,
            "exact_fallback_count": exact_fallback_count,
            "exact_fallback_rate": exact_fallback_count / total_unsupported if total_unsupported > 0 else 0.0,
            "clean_fallback_count": clean_fallback_count,
            "clean_fallback_rate": clean_fallback_count / total_unsupported if total_unsupported > 0 else 0.0,
        },
        "generation_errors": generation_error_count,
        "generation_failures": generation_failures,
    }


def parse_human_eval_csv(
    csv_path: str,
    supported_question_ids: List[str],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse and strictly validate manual_review.csv human scoring input."""
    if not os.path.exists(csv_path):
        return None, ["manual_review.csv file missing."]

    errors = []
    rows = []

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        return None, [f"Failed to read CSV: {e}"]

    seen_ids = set()
    scores = {}

    for row in rows:
        q_id = row.get("question_id", "").strip()
        if not q_id:
            errors.append("CSV row missing question_id.")
            continue

        if q_id in seen_ids:
            errors.append(f"Duplicate question_id '{q_id}' in CSV.")
            continue
        seen_ids.add(q_id)

        if q_id not in supported_question_ids:
            errors.append(f"Unknown question_id '{q_id}' in CSV.")
            continue

        raw_c = row.get("correctness", "").strip()
        raw_g = row.get("groundedness", "").strip()
        raw_s = row.get("citation_support", "").strip()

        # Check for blank / pending human scores
        if not raw_c or not raw_g or not raw_s:
            errors.append(f"Missing human score values for {q_id} (found c='{raw_c}', g='{raw_g}', s='{raw_s}').")
            continue

        try:
            val_c = int(raw_c)
            val_g = int(raw_g)
            val_s = int(raw_s)

            if val_c not in (0, 1, 2) or val_g not in (0, 1, 2) or val_s not in (0, 1, 2):
                errors.append(f"Out of range score for {q_id}: must be 0, 1, or 2.")
                continue

            scores[q_id] = {
                "correctness": val_c,
                "groundedness": val_g,
                "citation_support": val_s,
                "notes": row.get("notes", "").strip(),
            }
        except ValueError:
            errors.append(f"Non-integer score value for {q_id}.")

    missing_ids = set(supported_question_ids) - set(scores.keys())
    if missing_ids:
        errors.append(f"Human review incomplete for IDs: {sorted(list(missing_ids))}")

    if errors:
        return None, errors

    # Compute human evaluation summary
    total = len(scores)
    fully_correct = sum(1 for s in scores.values() if s["correctness"] == 2)
    fully_grounded = sum(1 for s in scores.values() if s["groundedness"] == 2)
    fully_cited = sum(1 for s in scores.values() if s["citation_support"] == 2)

    mean_c = sum(s["correctness"] for s in scores.values()) / total
    mean_g = sum(s["groundedness"] for s in scores.values()) / total
    mean_s = sum(s["citation_support"] for s in scores.values()) / total

    return {
        "status": "COMPLETED",
        "total_reviewed": total,
        "fully_correct_count": fully_correct,
        "fully_correct_rate": fully_correct / total,
        "mean_correctness": mean_c,
        "fully_grounded_count": fully_grounded,
        "fully_grounded_rate": fully_grounded / total,
        "mean_groundedness": mean_g,
        "fully_citation_supported_count": fully_cited,
        "fully_citation_supported_rate": fully_cited / total,
        "mean_citation_support": mean_s,
        "per_question_scores": scores,
    }, []


def generate_manual_review_csv_template(
    golden_questions: List[Dict[str, Any]],
    rag_results: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
    output_path: str = "evaluation/results/manual_review.csv",
) -> None:
    """Generate manual_review.csv template file with blank human scoring columns."""
    supported_golds = [q for q in golden_questions if q["type"] == "supported"]
    rag_map = {r["question_id"]: r for r in rag_results}
    ret_map = {r["question_id"]: r for r in retrieval_results}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "question",
            "reference_points",
            "rendered_answer",
            "citation_ids",
            "cited_filenames_pages",
            "retrieved_top4_chunks",
            "correctness",
            "groundedness",
            "citation_support",
            "notes",
        ])

        for q in supported_golds:
            q_id = q["id"]
            rag_r = rag_map.get(q_id, {})
            ret_r = ret_map.get(q_id, {})

            ref_pts = " | ".join(q.get("reference_points", []))
            rend_ans = rag_r.get("rendered_answer", "") or rag_r.get("error", "FAILED")
            c_ids = ", ".join(rag_r.get("citation_ids", []))

            cited_meta = []
            for c in rag_r.get("citations", []):
                cited_meta.append(f"{c['filename']} p.{c['page_number']}")
            cited_str = "; ".join(cited_meta)

            ret_meta = []
            for c in ret_r.get("retrieved_chunks", []):
                ret_meta.append(f"#{c['rank']} {c['filename']} p.{c['page_number']} ({c['chunk_id']})")
            ret_str = "; ".join(ret_meta)

            writer.writerow([
                q_id,
                q["question"],
                ref_pts,
                rend_ans,
                c_ids,
                cited_str,
                ret_str,
                "",  # correctness (0/1/2) - left blank for human reviewer
                "",  # groundedness (0/1/2) - left blank for human reviewer
                "",  # citation_support (0/1/2) - left blank for human reviewer
                "",  # notes - left blank for human reviewer
            ])


def run_scorer(
    results_dir: str = "evaluation/results",
    golden_path: str = "evaluation/golden_questions.json",
) -> Dict[str, Any]:
    """Execute complete scoring workflow and write summary.json."""
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"Golden questions missing: {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    golds = golden_data["questions"]

    ret_path = os.path.join(results_dir, "retrieval_results.json")
    rag_path = os.path.join(results_dir, "rag_results.json")

    if not os.path.exists(ret_path) or not os.path.exists(rag_path):
        raise FileNotFoundError("Retrieval or RAG evaluation results missing.")

    with open(ret_path, "r", encoding="utf-8") as f:
        ret_results = json.load(f)

    with open(rag_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    ret_scores = score_retrieval(ret_results, golds)
    rag_scores = score_rag_generation(rag_results, golds)

    # Ensure manual_review.csv template exists
    csv_path = os.path.join(results_dir, "manual_review.csv")
    if not os.path.exists(csv_path):
        generate_manual_review_csv_template(golds, rag_results, ret_results, csv_path)

    supported_ids = [q["id"] for q in golds if q["type"] == "supported"]
    human_scores, human_errors = parse_human_eval_csv(csv_path, supported_ids)

    human_status = "PENDING_HUMAN_REVIEW"
    if human_scores:
        human_status = "COMPLETED"

    summary = {
        "scoring_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "golden_questions_count": len(golds),
        "retrieval_metrics": ret_scores,
        "rag_generation_metrics": rag_scores,
        "human_evaluation": {
            "status": human_status,
            "validation_errors": human_errors if not human_scores else [],
            "scores": human_scores,
        }
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Scoring completed! Summary saved to '{summary_path}'.")
    return summary


def main() -> None:
    """CLI entrypoint for scorer."""
    parser = argparse.ArgumentParser(description="Stage 14 Evaluation Scorer")
    parser.add_argument("--results", type=str, default="evaluation/results")
    parser.add_argument("--golden", type=str, default="evaluation/golden_questions.json")
    args = parser.parse_args()

    try:
        run_scorer(results_dir=args.results, golden_path=args.golden)
    except Exception as e:
        print(f"ERROR during scoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
