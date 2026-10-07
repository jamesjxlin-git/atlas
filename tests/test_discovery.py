import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from src.atlas.discovery import credibility_reasons, select_readings, PaperDiscovery
from src.atlas.metadata import MetadataClient, MetadataError
from src.atlas.scholar import Paper, normalize_doi, reconstruct_abstract, paper_from_openalex, paper_from_crossref, safe_url, reference_ris


def paper(**kwargs):
    values = dict(id="https://openalex.org/W1", title="Research evidence", provider="OpenAlex",
                  metadata_url="https://openalex.org/W1", fetched_at="2026-10-06T00:00:00+00:00",
                  doi="10.1234/example", year=2025, abstract="Research abstract")
    return Paper(**{**values, **kwargs})


def client(handler, **kwargs):
    return MetadataClient(client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def test_metadata_missing_is_distinct_from_zero_and_abstract_is_bounded():
    result = paper_from_openalex({"id":"W1", "title":"<i>Paper</i>", "authorships":[],
                                 "abstract_inverted_index":{"world":[1],"Hello":[0],"bad":[1000000000]}}, "now")
    assert result.title == "Paper" and result.abstract == "Hello world"
    assert result.citation_count is None
    result = paper_from_crossref({"DOI":"10.1234/Test", "title":["Study"],
                                 "is-referenced-by-count":0, "abstract":"<jats:p>Text</jats:p>"}, "now")
    assert result.citation_count == 0 and result.abstract == "Text"
    assert normalize_doi("https://doi.org/10.1234/Test") == "10.1234/test"
    assert reconstruct_abstract(None) == ""


@pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///tmp/paper", "https://user:password@example.org", "https://[invalid", None])
def test_untrusted_link_schemes_are_not_rendered(url):
    assert safe_url(url) is None


def test_three_reasons_do_not_invent_missing_evidence_or_prestige():
    reasons = credibility_reasons(paper(doi=None))
    assert len(reasons) == 3 and all(r["label"] == "Evidence gap" for r in reasons)
    a = paper(journal="Journal", citation_count=10, affiliations=["Famous University"])
    b = replace(a, id="different", affiliations=["Regional College"])
    assert select_readings([(a, 1)], current_year=2026)[0].selection_score == select_readings([(b, 1)], current_year=2026)[0].selection_score
    assert len(credibility_reasons(a)) == 3
    assert "peer reviewed" not in json.dumps(credibility_reasons(a)).lower()
    assert reference_ris(a).startswith("TY  -")


@pytest.mark.parametrize("kind,status,excluded", [("retraction","retracted",True),("expression-of-concern","concern",True),("correction","correction_or_update",False)])
def test_checks_incoming_editorial_notice_not_original_record(kind,status,excluded):
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,json={"message":{"total-results":100,"items":[
            {"DOI":"10.1234/notice","title":["Notice"],"update-to":[{"DOI":"10.1234/example","type":kind,"source":"publisher"}]},
            {"DOI":"10.1234/other","update-to":[{"DOI":"10.1234/unrelated","type":"retraction"}]}
        ]}})
    result=client(handler).check_updates(paper())
    expected = "partial_check" if kind == "correction" else status
    assert result.integrity_status == expected
    assert len(result.notices) == 1 and result.notices[0]["doi"] == "10.1234/notice"
    assert requests[0].url.params["filter"] == "updates:10.1234/example"
    assert bool(select_readings([(result, 1)])) != excluded


def test_failed_update_check_is_unknown_and_error_hides_credentials(monkeypatch):
    monkeypatch.setattr("src.atlas.metadata.time.sleep", lambda _: None)
    metadata=client(lambda request:httpx.Response(401,json={"secret":"private"}),openalex_key="secret-value")
    with pytest.raises(MetadataError) as error:
        metadata.get_json("OpenAlex","/works")
    assert "secret-value" not in str(error.value) and "private" not in str(error.value)
    result=metadata.check_updates(paper())
    assert result.integrity_status == "check_failed" and result.integrity_checked_at is None


def test_cache_retains_fetch_time_and_excludes_credentials(tmp_path):
    seen=[]
    def handler(request):
        seen.append(request)
        return httpx.Response(200,json={"results":[]})
    metadata=client(handler,cache_dir=tmp_path,openalex_key="secret-key",contact_email="private@example.org")
    first=metadata.get_json("OpenAlex","/works",{"search":"sleep"})
    second=metadata.get_json("OpenAlex","/works",{"search":"sleep"})
    assert first == second and len(seen) == 1
    assert seen[0].headers["Authorization"] == "Bearer secret-key"
    metadata.get_json("Crossref","/works")
    assert seen[-1].url.params["mailto"] == "private@example.org"
    contents="".join(p.read_text() for p in tmp_path.glob('*.json'))
    assert "secret-key" not in contents and "private@example.org" not in contents


def test_transport_and_rate_limit_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("src.atlas.metadata.time.sleep",lambda _:None)
    calls=[]
    def handler(request):
        calls.append(request)
        if len(calls)==1:
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(429,json={}) if len(calls)==2 else httpx.Response(200,json={"results":[]})
    assert client(handler).get_json("OpenAlex","/works")[0] == {"results":[]}
    assert len(calls)==3


def test_openalex_fallback_does_not_fabricate_open_access():
    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(401,json={})
        return httpx.Response(200,json={"message":{"items":[{"DOI":"10.1234/test","title":["Research"]}]}})
    metadata=client(handler)
    papers,warnings=metadata.search("research")
    assert papers[0].provider == "Crossref" and warnings
    with pytest.raises(MetadataError,match="cannot be verified"):
        metadata.search("research",open_access=True)


def test_discovery_deduplicates_excludes_notices_and_retains_unknowns():
    class Metadata:
        def search(self,*args,**kwargs):
            return [paper(),paper(id="duplicate"),paper(id="bad",doi="10.1234/bad",retracted=True),
                    paper(id="concern",doi="10.1234/concern"),paper(id="unknown",doi="10.1234/unknown")],[]
        def check_many(self,papers):
            for p in papers:
                p.integrity_status="concern" if p.id=="concern" else "check_failed"
            return papers
        def citing_papers(self,p):
            raise MetadataError("unavailable")
    class Ranker:
        def rerank(self,query,chunks,k):
            return [(chunk,float(10-i)) for i,(chunk,_) in enumerate(chunks)]
    report=PaperDiscovery(Metadata(),Ranker()).recommend("topic",count=3)
    assert len(report["recommendations"]) == 2
    assert {r["paper"]["id"] for r in report["recommendations"]} == {"https://openalex.org/W1","unknown"}
    assert all(len(r["reasons"])==3 for r in report["recommendations"])
    assert report["checked_pool_count"]==3 and report["warnings"]


@pytest.mark.parametrize('model_kind',['embedding','reranking'])
def test_model_loading_failure_is_actionable_without_transport_secrets(monkeypatch,model_kind):
    from src.atlas.models import load_embedding_model, load_reranking_model
    def fail(*args,**kwargs):
        raise httpx.ProxyError('transport secret-url-token')
    class_name='SentenceTransformer' if model_kind=='embedding' else 'CrossEncoder'
    monkeypatch.setattr('sentence_transformers.'+class_name,fail)
    loader=load_embedding_model if model_kind=='embedding' else load_reranking_model
    with pytest.raises(RuntimeError,match='HF_HUB_OFFLINE') as error:
        loader('model')
    assert 'secret-url-token' not in str(error.value)
