"""Explicit human relevance labels; similarity scores are never ground truth."""

import hashlib
import math

LABELS = {3: "Perfect", 2: "Close", 1: "Decent", 0: "Bad"}
RUBRIC = {
    "Perfect": "Directly addresses the question with enough evidence for the requested answer.",
    "Close": "Useful evidence for the question, but incomplete or missing an important qualification.",
    "Decent": "Related background that does not directly answer the question.",
    "Bad": "Irrelevant, wrong scope, or misleading evidence for the question."
}


def chunk_id(chunk):
    body = f"{chunk.source}\0{chunk.start}\0{chunk.end}\0{chunk.text}"
    return hashlib.sha256(body.encode()).hexdigest()[:24]


def validate_labels(labels):
    if not isinstance(labels, dict):
        raise ValueError("labels must map chunk IDs to reviewed grades.")
    for identifier, item in labels.items():
        if not isinstance(identifier, str) or not isinstance(item, dict):
            raise ValueError("Each label requires a chunk ID and a label object.")
        grade = item.get("grade")
        if type(grade) is not int or grade not in LABELS:
            raise ValueError("Grades must be integers 0=Bad, 1=Decent, 2=Close, 3=Perfect.")
        if not isinstance(item.get("rationale", ""), str):
            raise ValueError("Review rationale must be text.")


def grade_results(results, labels, k=3):
    validate_labels(labels)
    rows = []
    for rank, (chunk, score) in enumerate(results[:k], 1):
        identifier = chunk_id(chunk)
        judgment = labels.get(identifier, {})
        grade = judgment.get("grade")
        rows.append({
            "rank": rank, "chunk_id": identifier, "grade": grade,
            "match": LABELS.get(grade, "Unjudged"),
            "rationale": judgment.get("rationale", ""),
            "source": chunk.source, "start": chunk.start, "end": chunk.end,
            "text": chunk.text, "ranking_score": score
        })
    return rows


def reviewed_hit_at_k(results, labels, k=3, *, minimum_grade=3):
    rows = grade_results(results, labels, k)
    if any(row["grade"] is not None and row["grade"] >= minimum_grade for row in rows):
        return 1
    if any(row["grade"] is None for row in rows):
        return None
    return 0


def best_match_at_k(results, labels, k=3):
    rows = grade_results(results, labels, k)
    known = [row["grade"] for row in rows if row["grade"] is not None]
    if 3 in known:
        return "Perfect"
    if any(row["grade"] is None for row in rows):
        return "Unjudged"
    return LABELS[max(known)] if known else "No results"


def ndcg_at_k(results, labels, k=3):
    rows = grade_results(results, labels, k)
    if any(row["grade"] is None for row in rows):
        return None
    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
    ideal = sorted((item["grade"] for item in labels.values()), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg([row["grade"] for row in rows]) / ideal_score if ideal_score else None


def graded_diagnostics(results, labels, *, label_scope="judged_pool"):
    return {
        "top_3": grade_results(results, labels, 3),
        "hit_at_1": reviewed_hit_at_k(results, labels, 1),
        "hit_at_3": reviewed_hit_at_k(results, labels, 3),
        "useful_hit_at_3": reviewed_hit_at_k(results, labels, 3, minimum_grade=2),
        "best_match_at_1": best_match_at_k(results, labels, 1),
        "best_match_at_3": best_match_at_k(results, labels, 3),
        "ndcg_at_3": ndcg_at_k(results, labels, 3),
        "ndcg_ideal_scope": label_scope,
        "judged_top_3": sum(row["grade"] is not None for row in grade_results(results, labels, 3)),
        "top_3_count": min(3, len(results))
    }
