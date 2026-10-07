from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from test_rag import CharacterEncoding, EmbeddingStub, RerankingStub

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    # The UI exercises real services with deterministic local inference stubs.
    monkeypatch.setattr('sentence_transformers.SentenceTransformer',lambda *a,**k:EmbeddingStub())
    monkeypatch.setattr('sentence_transformers.CrossEncoder',lambda *a,**k:RerankingStub())
    # Avoid tiktoken's first-use encoding download so the suite runs offline.
    monkeypatch.setattr('tiktoken.encoding_for_model',lambda *a,**k:CharacterEncoding())
    import streamlit as st
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()


def test_web_initial_discovery_has_no_loading_or_generation_errors(app):
    assert not app.exception
    assert app.sidebar.radio(key='page').value=='Discover'
    assert app.text_input[0].label=='Research topic'


def test_web_read_preview_review_and_switch_clear_session(app):
    app.sidebar.radio(key='page').set_value('Read & ask').run()
    app.radio(key='paper_source').set_value('Try the synthetic example').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['active_upload']['filename']=='sleep_study.txt'
    app.text_area[0].set_value('How many students were recruited?')
    response=next(widget for widget in app.radio if widget.label=='Response')
    response.set_value('Preview source evidence')
    ask=next(widget for widget in app.button if widget.label=='Ask Atlas')
    ask.click().run()
    assert not app.exception and len(app.session_state['history'])==1
    assert app.session_state['history'][0]['generation'] is None
    assert any(metric.value=='Unknown' for metric in app.metric)
    grade=next(widget for widget in app.selectbox if widget.label=='Relevance judgment')
    grade.set_value('Perfect').run()
    assert not app.exception
    assert next(metric for metric in app.metric if metric.label=='Hit@1 · Perfect evidence').value=='1'
    app.radio(key='paper_source').set_value('Upload my paper').run()
    assert not app.exception
    assert 'reader' not in app.session_state and 'history' not in app.session_state


def test_web_graded_evaluation_runs_and_displays_match_labels(app):
    app.sidebar.radio(key='page').set_value('Evaluate').run()
    next(widget for widget in app.button if widget.label=='Evaluate retrieval').click().run()
    assert not app.exception
    report=app.session_state['evaluation_report']
    assert report['dataset']=='Synthetic demonstration'
    assert report['summary']['answerable_cases']==3
    assert report['cases'][0]['reranked']['matches']['top_3'][0]['match']=='Perfect'
    assert len(app.metric)==6


def test_selecting_discovered_paper_starts_new_reading_session(app):
    from src.atlas.scholar import Paper
    from src.atlas.discovery import select_readings
    p=Paper(id='https://openalex.org/W1',title='Selected paper',provider='OpenAlex',metadata_url='https://openalex.org/W1',fetched_at='now',doi='10.1234/test')
    rec=select_readings([(p,1)])[0].to_dict()
    app.session_state['discovery_report']={'recommendations':[rec],'candidate_count':1,'checked_pool_count':1,'warnings':[],'excluded':[]}
    app.session_state['history']=[{'old':'session'}]
    app.run()
    assert not app.exception
    app.button(key='select_1').click().run()
    assert not app.exception
    assert app.sidebar.radio(key='page').value=='Read & ask'
    assert app.session_state['selected_paper']['title']=='Selected paper'
    assert 'history' not in app.session_state


def test_navigating_away_keeps_reading_session(app):
    app.sidebar.radio(key='page').set_value('Read & ask').run()
    app.radio(key='paper_source').set_value('Try the synthetic example').run()
    next(widget for widget in app.button if widget.label=='Load paper').click().run()
    app.sidebar.radio(key='page').set_value('Discover').run()
    app.sidebar.radio(key='page').set_value('Read & ask').run()
    assert not app.exception
    assert app.session_state['active_upload']['filename']=='sleep_study.txt'
