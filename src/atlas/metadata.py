"""Bounded HTTP calls, metadata caching, and editorial-update checks."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote

import httpx

from .scholar import normalize_doi, paper_from_openalex, paper_from_crossref, safe_url


class MetadataError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class MetadataClient:
    def __init__(self, *, client=None, cache_dir=None, openalex_key=None, contact_email=None):
        self.client = client or httpx.Client(
            timeout=20, follow_redirects=False,
            headers={"User-Agent": "AtlasResearchTutor/0.2"}
        )
        self.owns_client = client is None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.openalex_key = openalex_key
        self.contact_email = contact_email

    def close(self):
        if self.owns_client:
            self.client.close()

    def get_json(self, provider, path, params=None, *, ttl_hours=6):
        params = dict(params or {})
        hosts = {"OpenAlex": "https://api.openalex.org", "Crossref": "https://api.crossref.org"}
        if provider not in hosts or not path.startswith("/") or "://" in path:
            raise ValueError("Unknown metadata provider or endpoint.")
        # Credentials are never serialized into cache keys, files, or errors.
        key = hashlib.sha256(json.dumps([provider, path, params], sort_keys=True).encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache and cache.exists():
            try:
                record = json.loads(cache.read_text())
                fetched = datetime.fromisoformat(record["fetched_at"])
                age = (datetime.now(timezone.utc) - fetched).total_seconds()
                if 0 <= age < ttl_hours * 3600:
                    return record["data"], record["fetched_at"]
            except (ValueError, KeyError, TypeError, OSError):
                pass
        headers = {}
        if provider == "OpenAlex" and self.openalex_key:
            headers["Authorization"] = f"Bearer {self.openalex_key}"
        if provider == "Crossref" and self.contact_email:
            params["mailto"] = self.contact_email
        for attempt in range(3):
            try:
                response = self.client.get(hosts[provider] + path, params=params, headers=headers)
            except httpx.TransportError:
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise MetadataError(f"{provider} metadata request failed. Retry later or check API access.") from None
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
            try:
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Expected a metadata JSON object.")
            except (httpx.HTTPError, ValueError):
                raise MetadataError(f"{provider} metadata request failed. Retry later or check API access.") from None
            break
        fetched_at = utc_now()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False, encoding="utf-8") as file:
                json.dump({"fetched_at": fetched_at, "data": data}, file)
                temporary = Path(file.name)
            try:
                os.replace(temporary, cache)
            finally:
                temporary.unlink(missing_ok=True)
        return data, fetched_at

    def search(self, topic, *, limit=30, since_year=None, open_access=False):
        filters = ["is_retracted:false"]
        if since_year:
            filters.append(f"from_publication_date:{since_year}-01-01")
        if open_access:
            filters.append("is_oa:true")
        try:
            data, fetched = self.get_json("OpenAlex", "/works", {
                "search": topic, "per_page": limit, "filter": ",".join(filters)
            })
            papers = [paper_from_openalex(work, fetched) for work in data.get("results", [])]
            return papers, []
        except MetadataError:
            if open_access:
                raise MetadataError("OpenAlex is unavailable. The open-access filter cannot be verified.") from None
            params = {"query.bibliographic": topic, "rows": limit}
            if since_year:
                params["filter"] = f"from-pub-date:{since_year}-01-01"
            data, fetched = self.get_json("Crossref", "/works", params)
            papers = [paper_from_crossref(work, fetched)
                      for work in data.get("message", {}).get("items", [])]
            return papers, ["OpenAlex was unavailable; showing Crossref metadata. "
                            "Abstracts and affiliation coverage may be limited."]

    def check_updates(self, paper):
        if paper.retracted:
            paper.integrity_status = "retracted"
            return paper
        if not paper.doi:
            paper.integrity_status = "no_doi"
            return paper
        paper.integrity_url = "https://api.crossref.org/works?filter=updates:" + quote(paper.doi, safe="")
        try:
            data, fetched = self.get_json("Crossref", "/works", {
                "filter": f"updates:{paper.doi}", "rows": 50
            })
        except MetadataError:
            paper.integrity_status = "check_failed"
            return paper
        paper.integrity_checked_at = fetched
        notices = []
        for notice in data.get("message", {}).get("items", []):
            for update in notice.get("update-to", []):
                if normalize_doi(update.get("DOI")) == paper.doi:
                    notices.append({
                        "type": str(update.get("type", "update")).lower(),
                        "source": update.get("source", "Crossref"),
                        "doi": normalize_doi(notice.get("DOI")),
                        "url": safe_url(notice.get("URL")),
                        "title": (notice.get("title") or ["Editorial update"])[0]
                    })
        paper.notices = notices
        types = {notice["type"] for notice in notices}
        if any("retract" in kind for kind in types):
            paper.retracted = True
            paper.integrity_status = "retracted"
        elif any("concern" in kind for kind in types):
            paper.integrity_status = "concern"
        elif notices:
            paper.integrity_status = "correction_or_update"
        else:
            paper.integrity_status = "no_notice_found"
        if data.get("message", {}).get("total-results", 0) > 50 and paper.integrity_status not in {"retracted", "concern"}:
            paper.integrity_status = "partial_check"
        return paper

    def check_many(self, papers):
        with ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(self.check_updates, papers))

    def citing_papers(self, paper, limit=3):
        if paper.provider != "OpenAlex" or not paper.id:
            return []
        identifier = paper.id.rsplit("/", 1)[-1]
        data, _ = self.get_json("OpenAlex", "/works", {
            "filter": f"cites:{identifier}", "per_page": limit,
            "sort": "publication_date:desc"
        })
        return [
            {"title": work.get("title") or work.get("display_name"),
             "year": work.get("publication_year"), "url": safe_url(work.get("doi") or work.get("id"))}
            for work in data.get("results", [])
        ]
