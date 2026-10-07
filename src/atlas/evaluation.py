"""Binary retrieval checks, reviewed match grades, and separate generation diagnostics."""

import json
from pathlib import Path
from statistics import mean

from .generation import normalize_quote
from .grades import (
    LABELS, chunk_id, validate_labels, graded_diagnostics,
    reviewed_hit_at_k, grade_results
)


def hit_at_k(results, expected_text: str, k: int = 3) -> int:
    if k <= 0 or not expected_text.strip():
        raise ValueError("Provide positive k and nonempty expected text.")
    expected = normalize_quote(expected_text).casefold()
    return int(any(expected in normalize_quote(piece.text).casefold()
                   for piece, _ in results[:k]))


def reciprocal_rank(results, expected_text: str) -> float:
    for rank in range(1, len(results) + 1):
        if hit_at_k(results[rank - 1:rank], expected_text, 1):
            return 1.0 / rank
    return 0.0


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a nonempty JSON list.")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("Each case requires a nonempty query string.")
        if not isinstance(case.get("answerable", True), bool):
            raise ValueError("answerable must be a boolean.")
        if "labels" in case:
            validate_labels(case["labels"])
            if case.get("label_scope", "judged_pool") not in {"judged_pool", "complete_corpus"}:
                raise ValueError("label_scope must be judged_pool or complete_corpus.")
        elif case.get("answerable", True):
            if not isinstance(case.get("expected"), str) or not case["expected"].strip():
                raise ValueError("Answerable cases require expected evidence text or reviewed labels.")
        if not isinstance(case.get("answer_contains", []), list) or any(
                not isinstance(term, str) or not term.strip()
                for term in case.get("answer_contains", [])):
            raise ValueError("answer_contains must contain nonempty strings.")
    return cases


def load_cases(path):
    return validate_cases(json.loads(Path(path).read_text(encoding="utf-8")))


def reviewed_reciprocal_rank(results, labels):
    rows = grade_results(results, labels, len(results))
    for row in rows:
        if row["grade"] == 3:
            # Earlier unjudged passages might have been Perfect, making MRR unknown.
            return 1.0 / row["rank"] if all(
                earlier["grade"] is not None for earlier in rows[:row["rank"] - 1]
            ) else None
    return None if any(row["grade"] is None for row in rows) else 0.0


def evaluate(atlas, cases, *, generate=False):
    validate_cases(cases)
    rows = []
    universe = {chunk_id(chunk) for chunk in atlas.retriever.chunks}
    for case in cases:
        if "labels" in case and not set(case["labels"]).issubset(universe):
            raise ValueError("Reviewed labels contain unknown chunk IDs. Use the same document and chunk configuration as the review.")
        if case.get("label_scope") == "complete_corpus" and not universe.issubset(case.get("labels", {})):
            raise ValueError("Complete-corpus labels do not cover the current chunks. Rebuild/review labels after changing chunking.")
        result = atlas.search(case["query"])
        labels = case.get("labels", {})
        row = {"query": case["query"], "answerable": case.get("answerable", True),
               "label_method": "reviewed_grades" if "labels" in case else "expected_text"}
        for name, results in (("dense", result.dense), ("reranked", result.reranked)):
            diagnostics = graded_diagnostics(
                results, labels, label_scope=case.get("label_scope", "judged_pool")
            )
            metrics = {}
            if row["answerable"]:
                if "labels" in case:
                    metrics = {
                        "hit_at_1": diagnostics["hit_at_1"], "hit_at_3": diagnostics["hit_at_3"],
                        "reciprocal_rank": reviewed_reciprocal_rank(results, labels)
                    }
                else:
                    metrics = {
                        "hit_at_1": hit_at_k(results, case["expected"], 1),
                        "hit_at_3": hit_at_k(results, case["expected"], 3),
                        "reciprocal_rank": reciprocal_rank(results, case["expected"])
                    }
            row[name] = {**metrics, "matches": diagnostics}
        if row["answerable"]:
            context = [(p.chunk, p.score) for p in result.passages]
            row["context_hit"] = (
                reviewed_hit_at_k(context, labels, max(1, len(context))) if "labels" in case
                else hit_at_k(context, case["expected"], max(1, len(context)))
            )
        if generate:
            atlas.generate(result)
            generation = result.generation
            row["generation_status"] = generation.status
            terms = case.get("answer_contains", [])
            row["answer_term_match"] = (
                int(generation.status == "answered" and all(
                    term.casefold() in generation.answer.casefold() for term in terms
                )) if terms and row["answerable"] else None
            )
            row["correct_abstention"] = (
                int(generation.status == "insufficient_evidence") if not row["answerable"] else None
            )
        row["result"] = result.to_dict()
        rows.append(row)
    positives = [row for row in rows if row["answerable"]]
    def average(values):
        known = [value for value in values if value is not None]
        return mean(known) if known else None
    summary = {
        "cases": len(rows), "answerable_cases": len(positives),
        "unanswerable_cases": len(rows) - len(positives),
        "candidate_k": atlas.config.candidate_k, "config": vars(atlas.config),
        "retrieval_label": "Reviewed Perfect-grade hits when labels exist; otherwise expected-text containment.",
        "label_method_counts": {method: sum(row["label_method"] == method for row in rows)
                               for method in ("reviewed_grades", "expected_text")},
        "generation_label": "Smoke diagnostics; not factual accuracy or semantic entailment.",
        "index_ms": atlas.index_ms
    }
    for name in ("dense", "reranked"):
        summary[name] = {
            metric: average([row[name][metric] for row in positives])
            for metric in ("hit_at_1", "hit_at_3", "reciprocal_rank")
        }
        summary[name]["metric_case_counts"] = {
            metric: sum(row[name][metric] is not None for row in positives)
            for metric in ("hit_at_1", "hit_at_3", "reciprocal_rank")
        }
        summary[name]["ndcg_at_3"] = average([
            row[name]["matches"]["ndcg_at_3"] for row in positives if row["label_method"] == "reviewed_grades"
        ])
        summary[name]["metric_case_counts"]["ndcg_at_3"] = sum(
            row[name]["matches"]["ndcg_at_3"] is not None and row["label_method"] == "reviewed_grades"
            for row in positives
        )
        summary[name]["useful_hit_at_3"] = average([
            row[name]["matches"]["useful_hit_at_3"] for row in positives
            if row["label_method"] == "reviewed_grades"
        ])
        summary[name]["best_at_3_counts"] = {
            label: sum(row[name]["matches"]["best_match_at_3"] == label for row in positives)
            for label in [*LABELS.values(), "Unjudged", "No results"]
        }
    summary["context_hit_rate"] = average([row["context_hit"] for row in positives])
    if generate:
        for metric in ("answer_term_match", "correct_abstention"):
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[metric] = {"value": mean(values) if values else None, "cases": len(values)}
        summary["generation_status_counts"] = {
            status: sum(row["generation_status"] == status for row in rows)
            for status in ("answered", "insufficient_evidence", "refused", "invalid_evidence")
        }
        summary["tokens"] = {
            token: sum(row["result"]["generation"]["usage"][token] for row in rows)
            for token in ("input_tokens", "output_tokens")
        }
    summary["mean_timings_ms"] = {
        stage: mean(row["result"]["timings_ms"][stage] for row in rows)
        for stage in rows[0]["result"]["timings_ms"]
    }
    return {"summary": summary, "cases": rows}
