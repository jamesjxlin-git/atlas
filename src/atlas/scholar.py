"""Scholarly metadata, public-source URLs, and reproducible citation exports."""

from dataclasses import dataclass, field, asdict
import html
import re
from urllib.parse import urlparse, unquote


def safe_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username:
            return None
    except ValueError:
        return None
    return value


def normalize_doi(value):
    if not isinstance(value, str):
        return None
    value = unquote(value.strip())
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I).strip().lower()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def plain_text(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def reconstruct_abstract(index):
    if not isinstance(index, dict):
        return ""
    positions = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int) and not isinstance(offset, bool) and 0 <= offset < 10000:
                positions[offset] = word
    return " ".join(positions[i] for i in sorted(positions))


@dataclass
class Paper:
    id: str
    title: str
    provider: str
    metadata_url: str
    fetched_at: str
    doi: str | None = None
    year: int | None = None
    publication_date: str | None = None
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    journal: str | None = None
    work_type: str | None = None
    abstract: str = ""
    citation_count: int | None = None
    citation_percentile: float | None = None
    oa_url: str | None = None
    landing_url: str | None = None
    study_designs: list[str] = field(default_factory=list)
    retracted: bool = False
    is_notice: bool = False
    integrity_status: str = "not_checked"
    integrity_checked_at: str | None = None
    integrity_url: str | None = None
    notices: list[dict] = field(default_factory=list)
    citing_examples: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @property
    def reference(self):
        authors = ", ".join(self.authors[:6]) or "Authors unavailable"
        if len(self.authors) > 6:
            authors += ", et al."
        venue = self.journal or "Venue unavailable"
        doi = f" https://doi.org/{self.doi}" if self.doi else ""
        return f"{authors}. {self.title}. {venue} ({self.year or 'year unavailable'}).{doi}"


def paper_from_openalex(data, fetched_at):
    location = data.get("primary_location") or {}
    source = location.get("source") or {}
    authorships = data.get("authorships") or []
    affiliations = sorted({
        item["display_name"] for author in authorships
        for item in (author.get("institutions") or [])
        if item.get("display_name")
    })
    percentile = (data.get("citation_normalized_percentile") or {}).get("value")
    if type(percentile) not in (float, int) or not 0 <= percentile <= 1:
        percentile = None
    oa = data.get("best_oa_location") or {}
    retracted = data.get("is_retracted") is True
    return Paper(
        id=data.get("id", ""), title=plain_text(data.get("title") or data.get("display_name")),
        provider="OpenAlex", metadata_url=data.get("id", ""), fetched_at=fetched_at,
        doi=normalize_doi(data.get("doi")), year=data.get("publication_year"),
        publication_date=data.get("publication_date"),
        authors=[a["author"]["display_name"] for a in authorships
                 if (a.get("author") or {}).get("display_name")],
        affiliations=affiliations, journal=source.get("display_name"),
        work_type=data.get("type"), abstract=reconstruct_abstract(data.get("abstract_inverted_index")),
        citation_count=data.get("cited_by_count") if type(data.get("cited_by_count")) is int and data["cited_by_count"] >= 0 else None,
        citation_percentile=percentile,
        oa_url=safe_url(oa.get("pdf_url") or oa.get("landing_page_url")
                        or (data.get("open_access") or {}).get("oa_url")),
        landing_url=safe_url(location.get("landing_page_url") or data.get("doi")),
        study_designs=[item["display_name"] for item in (data.get("study_designs") or [])
                       if item.get("display_name")],
        retracted=retracted, integrity_status="retracted" if retracted else "not_checked"
    )


def paper_from_crossref(data, fetched_at):
    doi = normalize_doi(data.get("DOI"))
    date = next((data[key].get("date-parts", [[]])[0]
                 for key in ["published", "published-online", "published-print", "issued"]
                 if data.get(key, {}).get("date-parts")), [])
    year = date[0] if date else None
    return Paper(
        id=f"https://doi.org/{doi}" if doi else "", title=plain_text((data.get("title") or [""])[0]),
        provider="Crossref", metadata_url=f"https://api.crossref.org/works/{doi}",
        fetched_at=fetched_at, doi=doi, year=year,
        publication_date="-".join(str(n).zfill(2) for n in date) if date else None,
        authors=[" ".join(str(a.get(part, "")) for part in ["given", "family"]).strip()
                 for a in data.get("author", []) if a.get("given") or a.get("family")],
        affiliations=sorted({a["name"] for author in data.get("author", [])
                             for a in author.get("affiliation", []) if a.get("name")}),
        journal=(data.get("container-title") or [None])[0], work_type=data.get("type"),
        abstract=plain_text(data.get("abstract")),
        citation_count=data.get("is-referenced-by-count")
                       if type(data.get("is-referenced-by-count")) is int and data["is-referenced-by-count"] >= 0 else None,
        landing_url=safe_url(data.get("URL")), is_notice=bool(data.get("update-to")),
    )


def reference_ris(paper):
    kind = "JOUR" if paper.work_type in {"article", "review", "journal-article"} else "GEN"
    def clean(text):
        return " ".join(str(text).splitlines())
    lines = [f"TY  - {kind}", f"TI  - {clean(paper.title)}"]
    lines += [f"AU  - {clean(author)}" for author in paper.authors]
    if paper.year:
        lines.append(f"PY  - {paper.year}")
    if paper.journal:
        lines.append(f"JO  - {clean(paper.journal)}")
    if paper.doi:
        lines.append(f"DO  - {paper.doi}")
    if paper.landing_url:
        lines.append(f"UR  - {paper.landing_url}")
    return "\n".join(lines + ["ER  -", ""])
