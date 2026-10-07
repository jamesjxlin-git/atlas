import math
from types import SimpleNamespace

import pytest

from src.atlas.chunking import Chunk
from src.atlas.evaluation import evaluate, load_cases
from src.atlas.grades import chunk_id, graded_diagnostics, ndcg_at_k, reviewed_hit_at_k, validate_labels
from src.atlas.ingestion import load_documents, parse_upload
from src.atlas.pipeline import Atlas
from src.atlas.retrieval import Retriever
from src.atlas.reranking import Reranker
from test_rag import EmbeddingStub, RerankingStub, CharacterEncoding


def ranking():
    chunks=[Chunk(name,'paper',i,i+len(name)) for i,name in enumerate(['direct answer','partial answer','background','unrelated'])]
    labels={chunk_id(c):{'grade':3-i,'rationale':c.text} for i,c in enumerate(chunks)}
    return [(c,0.5) for c in chunks],labels


def test_perfect_hits_close_useful_hits_and_graded_ranking():
    results,labels=ranking()
    result=graded_diagnostics([results[1],results[0],results[2],results[3]],labels)
    assert result['hit_at_1']==0 and result['hit_at_3']==1
    assert result['best_match_at_1']=='Close' and result['best_match_at_3']=='Perfect'
    assert result['useful_hit_at_3']==1
    expected=(3+7/math.log2(3)+1/math.log2(4))/(7+3/math.log2(3)+1/math.log2(4))
    assert result['ndcg_at_3']==pytest.approx(expected)
    assert ndcg_at_k(results,labels)==pytest.approx(1)


def test_unjudged_is_unknown_even_when_scores_are_high():
    results,labels=ranking()
    partial={chunk_id(results[1][0]):labels[chunk_id(results[1][0])]}
    diagnostics=graded_diagnostics(results,partial)
    assert diagnostics['hit_at_3'] is None and diagnostics['ndcg_at_3'] is None
    assert diagnostics['best_match_at_3']=='Unjudged'
    # A reviewed Perfect passage proves a hit even when its neighbors are unjudged.
    perfect={chunk_id(results[0][0]):labels[chunk_id(results[0][0])]}
    assert reviewed_hit_at_k(results,perfect,3)==1


@pytest.mark.parametrize('grade',[True,4,-1,'Perfect',3.0])
def test_invalid_review_grades_are_rejected(grade):
    with pytest.raises(ValueError):
        validate_labels({'id':{'grade':grade}})


def test_synthetic_fixture_labels_cover_exact_current_chunks():
    atlas=Atlas(retriever=Retriever(model=EmbeddingStub()),reranker=Reranker(model=RerankingStub()),encoding=CharacterEncoding())
    atlas.index_documents(load_documents('data/graded_demo'))
    cases=load_cases('data/evaluation/graded_cases.json')
    assert len(atlas.retriever.chunks)==5
    assert {chunk_id(c) for c in atlas.retriever.chunks}==set(cases[0]['labels'])
    report=evaluate(atlas,cases)
    assert report['summary']['answerable_cases']==3
    assert report['summary']['reranked']['metric_case_counts']['hit_at_3']==3
    cases[0]['labels'].pop(next(iter(cases[0]['labels'])))
    with pytest.raises(ValueError,match='cover the current chunks'):
        evaluate(atlas,cases)


def test_text_upload_keeps_source_hash_and_bounds():
    result=parse_upload('../../paper.txt',b'Original text.\n')
    assert result.filename=='paper.txt' and result.documents[0].source=='paper.txt'
    assert result.documents[0].text=='Original text.\n' and len(result.fingerprint)==64
    for filename,payload,kwargs in [('file.exe',b'hello',{}),('x.pdf',b'fake pdf',{}),('x.txt',b'\xff',{}),('x.txt',b'abcd',{'max_bytes':3}),('x.txt',b'abcd',{'max_characters':3})]:
        with pytest.raises(ValueError):
            parse_upload(filename,payload,**kwargs)
