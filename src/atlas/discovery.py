"""A transparent reading shortlist from verified bibliographic metadata."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
import re

from .chunking import Chunk
from .scholar import Paper
from .metadata import MetadataError

READING_WEIGHTS = {"balanced": (0.80, 0.10, 0.10),
                   "foundational": (0.80, 0.20, 0.00),
                   "recent": (0.80, 0.00, 0.20)}


@dataclass
class Recommendation:
    paper: Paper
    relevance_rank: int
    relevance_score: float
    selection_score: float
    reasons: list[dict]
    caveats: list[str]

    def to_dict(self):
        return {**asdict(self), "reference": self.paper.reference}


def credibility_reasons(paper):
    reasons = []
    if paper.journal:
        reasons.append({
            "label": "Publication context",
            "text": f"Recorded in {paper.journal} as {paper.work_type or 'an unspecified work type'}.",
            "source_url": paper.metadata_url
        })
    if paper.citation_count is not None:
        detail = f"{paper.provider} reports {paper.citation_count:,} citing works."
        if paper.citation_percentile is not None:
            detail += f" Citation impact percentile: {100 * paper.citation_percentile:.1f}."
        reasons.append({"label": "Scholarly uptake", "text": detail, "source_url": paper.metadata_url})
    if paper.affiliations:
        reasons.append({
            "label": "Traceable affiliations",
            "text": "Author affiliations include " + "; ".join(paper.affiliations[:3]) + ".",
            "source_url": paper.metadata_url
        })
    if len(reasons) < 3 and paper.doi:
        reasons.append({"label": "Traceable publication",
                        "text": f"DOI {paper.doi} identifies a bibliographic record.",
                        "source_url": f"https://doi.org/{paper.doi}"})
    if len(reasons) < 3 and paper.study_designs:
        reasons.append({"label": "Reported study design",
                        "text": "OpenAlex automatically tags: " + ", ".join(paper.study_designs) + ".",
                        "source_url": paper.metadata_url})
    while len(reasons) < 3:
        reasons.append({"label": "Evidence gap",
                        "text": "Available metadata does not establish an additional credibility reason.",
                        "source_url": paper.metadata_url})
    return reasons[:3]


def paper_caveats(paper):
    caveats = ["Citation counts measure attention, not correctness or endorsement. "
               "Affiliations identify provenance, not study quality."]
    if paper.work_type in {"preprint", "posted-content"}:
        caveats.append("This is a preprint/posted manuscript; peer review is not established.")
    else:
        caveats.append("Publication metadata alone does not verify peer review or methodological quality.")
    if not paper.abstract:
        caveats.append("No abstract was available; topic ranking used the title.")
    if paper.study_designs:
        caveats.append("Study-design tags are automated metadata, not a critical appraisal.")
    status_text = {
        "no_notice_found": "No editorial notice found in the queried Crossref records as of the displayed check time; coverage can be incomplete.",
        "check_failed": "Editorial-update check failed; status remains unknown.",
        "not_checked": "Editorial-update status has not been checked.",
        "no_doi": "No DOI was available for an editorial-update check.",
        "correction_or_update": "An editorial correction/update is recorded; read the notice before citing.",
        "concern": "An expression of concern is recorded; excluded from the normal shortlist.",
        "partial_check": "Editorial-update results exceeded the check limit; review manually."
    }
    if paper.integrity_status in status_text:
        caveats.append(status_text[paper.integrity_status])
    return caveats


def select_readings(scored, *, count=5, preference="balanced", current_year=None):
    """Rank within a relevance-screened candidate pool; no prestige bonus."""
    if preference not in {"balanced", "foundational", "recent"}:
        raise ValueError("Unknown reading preference.")
    current_year = current_year or datetime.now(timezone.utc).year
    recommendations = []
    for rank, (paper, raw_score) in enumerate(scored, 1):
        if paper.retracted or paper.is_notice or paper.integrity_status == "concern":
            continue
        relevance = 1 / math.sqrt(rank)
        age = max(0, current_year - paper.year) if isinstance(paper.year, int) else None
        recent = 1 / (1 + age / 3) if age is not None else 0.5
        # Missing citations receive a neutral heuristic value, never a fabricated zero.
        uptake = paper.citation_percentile if paper.citation_percentile is not None else (
            min(1, math.log1p(max(0, paper.citation_count)) / math.log1p(1000))
            if paper.citation_count is not None else 0.5
        )
        weights = READING_WEIGHTS[preference]
        score = weights[0] * relevance + weights[1] * uptake + weights[2] * recent
        # Integrity unknown/correction remains visible; known retractions/concerns excluded.
        recommendations.append(Recommendation(
            paper, rank, float(raw_score), score,
            credibility_reasons(paper), paper_caveats(paper)
        ))
    recommendations.sort(key=lambda item: item.selection_score, reverse=True)
    return recommendations[:count]


class PaperDiscovery:
    def __init__(self, metadata, reranker):
        self.metadata = metadata
        self.reranker = reranker

    def recommend(self, topic, *, count=5, preference="balanced", since_year=None,
                  open_access=False, candidate_limit=30):
        if not topic.strip() or len(topic) > 500:
            raise ValueError("Enter a research topic between 1 and 500 characters.")
        if not 1 <= count <= 10 or not count <= candidate_limit <= 100:
            raise ValueError("Require 1 <= count <= 10 and count <= candidates <= 100.")
        if since_year is not None and not 1800 <= since_year <= datetime.now(timezone.utc).year:
            raise ValueError("Invalid publication year.")
        if preference not in {"balanced", "foundational", "recent"}:
            raise ValueError("Unknown reading preference.")
        papers, warnings = self.metadata.search(
            topic, limit=candidate_limit, since_year=since_year, open_access=open_access
        )
        # DOI first; a title/year fallback is used only for duplicate metadata records.
        unique = {}
        excluded = []
        for paper in papers:
            if (not paper.title or paper.retracted or paper.is_notice
                    or paper.work_type in {"retraction", "erratum", "paratext", "dataset", "software"}):
                excluded.append({"id": paper.id, "reason": "Ineligible work type, known retraction, editorial notice, or missing title"})
                continue
            fingerprint = paper.doi or re.sub(r"\W+", "", paper.title.casefold()) + str(paper.year)
            unique.setdefault(fingerprint, paper)
        chunks = []
        by_source = {}
        for i, paper in enumerate(unique.values()):
            source = f"paper:{i}"
            text = paper.title + "\n" + paper.abstract
            chunk = Chunk(text, source, 0, len(text))
            chunks.append((chunk, 0.0))
            by_source[source] = paper
        ranked = self.reranker.rerank(topic, chunks, k=max(1, len(chunks)))
        # Integrity-check a bounded relevance pool; metadata only, not full-text appraisal.
        pool = ranked[:max(12, count * 2)]
        checked = self.metadata.check_many([by_source[chunk.source] for chunk, _ in pool])
        scored = list(zip(checked, [score for _, score in pool]))
        for paper in checked:
            if paper.retracted or paper.integrity_status == "concern":
                excluded.append({"id": paper.id, "reason": paper.integrity_status})
        chosen = select_readings(scored, count=count, preference=preference)
        for recommendation in chosen:
            try:
                recommendation.paper.citing_examples = self.metadata.citing_papers(recommendation.paper)
            except MetadataError:
                warnings.append("Some citing-paper examples were unavailable.")
        return {
            "topic": topic, "preference": preference,
            "settings": {"policy_version": "atlas-shortlist-v1",
                         "reranking_model": getattr(self.reranker, "model_name", None),
                         "weights_relevance_uptake_freshness": READING_WEIGHTS[preference],
                         "candidate_limit": candidate_limit, "requested_count": count,
                         "since_year": since_year, "open_access_only": open_access},
            "candidate_count": len(papers), "checked_pool_count": len(pool),
            "recommendations": [item.to_dict() for item in chosen],
            "excluded": excluded, "warnings": list(dict.fromkeys(warnings)),
            "ranking_note": "Relevance-first heuristic shortlist from title/abstract metadata. "
                            "Selection scores are not probabilities of credibility.",
            "integrity_note": "Retractions and concerns found in available metadata are excluded; "
                              "missing notices do not certify correctness."
        }
