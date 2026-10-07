import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import OpenAI
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from src.atlas.chunking import Chunk
from src.atlas.config import AtlasConfig
from src.atlas.context import Passage, pack_context, serialize_context
from src.atlas.evaluation import evaluate, hit_at_k, load_cases
from src.atlas.generation import (
    Claim, Evidence, Generator, GenerationError, GroundedAnswer, validate_evidence
)
from src.atlas.ingestion import Document, load_documents
from src.atlas.pipeline import Atlas
from src.atlas.retrieval import Retriever
from src.atlas.reranking import Reranker


class CharacterEncoding:
    def encode(self, text, **kwargs):
        return list(text)


class EmbeddingStub:
    max_seq_length = 256

    def __init__(self):
        self.calls = 0
        backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    def encode_document(self, texts, **kwargs):
        self.calls += 1
        return np.asarray([[1.0, 0.0] if "500" in text else [0.0, 1.0] for text in texts])

    def encode_query(self, text, **kwargs):
        return np.array([1.0, 0.0])


class RerankingStub:
    def predict(self, pairs, **kwargs):
        return np.array([1.0 if "500" in text else -1.0 for _, text in pairs])


def evidence_passage():
    text = "We recruited 500 university students."
    return Passage(1, Chunk(text, "study.txt", 0, len(text)), 1.0)


def supported_answer():
    return GroundedAnswer(
        answerable=True, reason="", claims=[
            Claim(text="The study recruited 500 university students.",
                  evidence=[Evidence(passage_id=1, quote="We recruited 500 university students.")])
        ]
    )


def response_json(answer=None, *, status="completed", refusal=False):
    content = (
        [{"type": "refusal", "refusal": "Request refused."}] if refusal else
        [{"type": "output_text", "text": answer.model_dump_json(), "annotations": []}]
    )
    return {
        "id": "resp_test", "object": "response", "created_at": 1,
        "status": status, "model": "gpt-4.1-mini-2025-04-14",
        "error": None, "incomplete_details": None,
        "instructions": None, "metadata": {}, "parallel_tool_calls": False,
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": content}],
        "tools": [], "tool_choice": "auto", "temperature": 1, "top_p": 1,
        "usage": {"input_tokens": 100, "output_tokens": 30, "total_tokens": 130,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}}
    }


def sdk_client(payload, captured=None):
    def handler(request):
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(200, json=payload)
    return OpenAI(api_key="test-only-key", max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_real_sdk_parse_and_citations():
    captured = []
    generator = Generator(client=sdk_client(response_json(supported_answer()), captured))
    result = generator.generate("How many students?", [evidence_passage()])
    assert result.status == "answered"
    assert result.answer == "The study recruited 500 university students. [1]"
    assert result.input_tokens == 100
    request = captured[0]
    assert request["store"] is False
    schema = request["text"]["format"]
    assert schema["type"] == "json_schema" and schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert request["model"] == "gpt-4.1-mini-2025-04-14"


@pytest.mark.parametrize("mutation", ["unknown_id", "invented_quote", "empty_quote",
                                      "missing_evidence", "own_citation", "no_claims"])
def test_invalid_answer_is_withheld(mutation):
    answer = supported_answer()
    if mutation == "unknown_id":
        answer.claims[0].evidence[0].passage_id = 99
    elif mutation == "invented_quote":
        answer.claims[0].evidence[0].quote = "We recruited 600 students in London."
    elif mutation == "empty_quote":
        answer.claims[0].evidence[0].quote = ""
    elif mutation == "missing_evidence":
        answer.claims[0].evidence = []
    elif mutation == "own_citation":
        answer.claims[0].text += " [99]"
    else:
        answer.claims = []
    result = Generator(client=sdk_client(response_json(answer))).generate(
        "How many?", [evidence_passage()]
    )
    assert result.status == "invalid_evidence"
    assert result.claims == []
    assert "withheld" in result.answer


def test_pdf_whitespace_quote_normalization():
    passage = Passage(1, Chunk("We recruited\n500 university students.", "paper", 0, 36), 1)
    assert validate_evidence(supported_answer(), [passage]) is None


def test_quote_check_is_not_entailment():
    answer = supported_answer()
    answer.claims[0].text = "The study recruited 600 students."
    # This limitation is deliberate and documented: human semantic review is needed.
    assert validate_evidence(answer, [evidence_passage()]) is None


def test_unanswerable():
    answer = GroundedAnswer(answerable=False, claims=[], reason="No named university is given.")
    result = Generator(client=sdk_client(response_json(answer))).generate(
        "What university?", [evidence_passage()]
    )
    assert result.status == "insufficient_evidence"


def test_no_context_makes_no_api_call():
    client = SimpleNamespace(responses=None)
    assert Generator(client=client).generate("What?", []).status == "insufficient_evidence"


def test_provider_refusal_is_distinct():
    result = Generator(client=sdk_client(response_json(refusal=True))).generate(
        "What?", [evidence_passage()]
    )
    assert result.status == "refused"


def test_incomplete_response_not_shown():
    with pytest.raises(GenerationError, match="did not complete"):
        Generator(client=sdk_client(response_json(supported_answer(), status="incomplete"))).generate(
            "What?", [evidence_passage()]
        )


def test_api_failure_does_not_expose_provider_body():
    client = OpenAI(api_key="test-only-key", max_retries=0, http_client=httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            429, json={"error": {"message": "private test payload", "type": "rate_limit",
                                "code": "rate_limit", "param": None}}
        ))
    ))
    with pytest.raises(GenerationError) as error:
        Generator(client=client).generate("What?", [evidence_passage()])
    assert "RateLimitError" in str(error.value)
    assert "private test payload" not in str(error.value)


def test_missing_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="Set OPENAI_API_KEY"):
        Generator().generate("What?", [evidence_passage()])


def test_context_budget_duplicates_and_ids():
    small = evidence_passage().chunk
    large = Chunk("x" * 1000, "large", 0, 1000)
    other = Chunk("A separate passage with more detail.", "other", 0, 36)
    passages = pack_context([(large, 3), (small, 2), (small, 1), (other, 0)],
                            CharacterEncoding(), max_tokens=170, max_passages=3)
    assert [p.id for p in passages] == [1, 2]
    assert passages[0].chunk.source == "study.txt"
    assert len(serialize_context(passages)) <= 170


def test_empty_context_budget():
    assert pack_context([(evidence_passage().chunk, 1)], CharacterEncoding(), 1) == []


def test_cache_reuse_and_content_invalidation(tmp_path):
    model = EmbeddingStub()
    retriever = Retriever(model=model, cache_dir=tmp_path)
    chunks = [evidence_passage().chunk, Chunk("Other content", "other", 0, 13)]
    retriever.index(chunks)
    retriever.index(chunks)
    assert model.calls == 1
    assert retriever.search("students", 1)[0][0] == chunks[0]
    changed = [Chunk("500 revised students", "study", 0, 20)]
    retriever.index(changed)
    assert model.calls == 2


def test_corrupt_cache_rebuild(tmp_path):
    model = EmbeddingStub()
    retriever = Retriever(model=model, cache_dir=tmp_path)
    retriever.index([evidence_passage().chunk])
    next(tmp_path.glob("*.npy")).write_bytes(b"corrupted")
    retriever.index([evidence_passage().chunk])
    assert model.calls == 2


def test_model_name_invalidates_cache(tmp_path):
    first = EmbeddingStub()
    second = EmbeddingStub()
    Retriever(model_name="model-a", model=first, cache_dir=tmp_path).index([evidence_passage().chunk])
    Retriever(model_name="model-b", model=second, cache_dir=tmp_path).index([evidence_passage().chunk])
    assert first.calls == second.calls == 1
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_reranker_empty_and_invalid_scores():
    reranker = Reranker(model=RerankingStub())
    assert reranker.rerank("query", []) == []
    bad = SimpleNamespace(predict=lambda pairs, **kwargs: np.array([np.nan]))
    with pytest.raises(ValueError, match="Invalid reranker"):
        Reranker(model=bad).rerank("query", [(evidence_passage().chunk, 1)])


def test_text_ingestion_preserves_offsets_and_relative_sources(tmp_path):
    (tmp_path / "a.txt").write_text("First paper", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.md").write_text("Second paper", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert [d.source for d in documents] == ["a.txt", "sub/b.md"]
    assert documents[0].text == "First paper"


def test_ingestion_rejects_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="No readable"):
        load_documents(tmp_path)


def test_pdf_page_ingestion_and_scanned_pdf_error(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=400)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica")
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 30 300 Td (We recruited 500 students.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    file = tmp_path / "paper.pdf"
    writer.write(file)
    documents = load_documents(file)
    assert documents[0].source == "paper.pdf#page=1"
    assert "500 students" in documents[0].text
    empty = PdfWriter()
    empty.add_blank_page(width=400, height=400)
    empty.write(tmp_path / "scan.pdf")
    with pytest.raises(ValueError, match="OCR"):
        load_documents(tmp_path / "scan.pdf")


def test_pipeline_and_evaluation_with_real_sdk(tmp_path):
    atlas = Atlas(
        AtlasConfig(context_tokens=1000),
        retriever=Retriever(model=EmbeddingStub(), cache_dir=tmp_path),
        reranker=Reranker(model=RerankingStub()),
        generator=Generator(client=sdk_client(response_json(supported_answer()))),
        encoding=CharacterEncoding()
    )
    atlas.index_documents([Document("We recruited 500 university students.", "study.txt")])
    result = atlas.ask("How many students?")
    assert result.generation.status == "answered"
    assert result.passages[0].chunk.source == "study.txt"
    assert "generation" in result.timings_ms
    report = evaluate(atlas, [{"query": "How many?", "expected": "500 university students",
                              "answer_contains": ["500"]}], generate=True)
    assert report["summary"]["dense"]["hit_at_1"] == 1
    assert report["summary"]["answer_term_match"]["value"] == 1
    assert report["cases"][0]["result"]["generation"]["usage"]["input_tokens"] == 100


def test_no_positives_does_not_divide_by_zero(tmp_path):
    atlas = Atlas(
        retriever=Retriever(model=EmbeddingStub()), reranker=Reranker(model=RerankingStub()),
        encoding=CharacterEncoding()
    )
    atlas.index_documents([Document("We recruited 500 university students.", "study")])
    report = evaluate(atlas, [{"query": "Where?", "answerable": False}])
    assert report["summary"]["dense"]["hit_at_1"] is None


def test_empty_expected_rejected():
    with pytest.raises(ValueError):
        hit_at_k([(evidence_passage().chunk, 1)], "")


def test_invalid_eval_labels(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('[{"query": "What?", "answerable": "false"}]')
    with pytest.raises(ValueError, match="boolean"):
        load_cases(path)


def test_overview_reuses_original_evidence_and_tracks_four_requests():
    from src.atlas.overview import build_overview
    captured=[]
    generator=Generator(client=sdk_client(response_json(supported_answer()),captured))
    atlas=Atlas(retriever=Retriever(model=EmbeddingStub()),reranker=Reranker(model=RerankingStub()),generator=generator,encoding=CharacterEncoding())
    atlas.index_documents([Document(evidence_passage().chunk.text,'study.txt')])
    overview=build_overview(atlas)
    assert len(captured)==4 and len(overview['sections'])==4
    assert overview['tokens']=={'input_tokens':400,'output_tokens':120}
    assert all(s['result']['generation']['status']=='answered' for s in overview['sections'])
    assert all(s['result']['context'][0]['text']==evidence_passage().chunk.text for s in overview['sections'])
