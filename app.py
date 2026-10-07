"""Atlas local research workflow. Run: python -m streamlit run app.py"""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from src.atlas.config import AtlasConfig
from src.atlas.discovery import PaperDiscovery
from src.atlas.evaluation import evaluate, load_cases
from src.atlas.generation import GenerationError, Generator
from src.atlas.grades import RUBRIC, chunk_id, graded_diagnostics
from src.atlas.ingestion import parse_upload, load_documents
from src.atlas.metadata import MetadataClient, MetadataError
from src.atlas.models import load_embedding_model, load_reranking_model
from src.atlas.overview import build_overview, suggested_questions
from src.atlas.pipeline import Atlas
from src.atlas.retrieval import Retriever
from src.atlas.reranking import Reranker
from src.atlas.scholar import Paper, safe_url, reference_ris


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
st.set_page_config(page_title="Atlas | Research with evidence", page_icon="📚", layout="wide")


@st.cache_resource
def embedding_model():
    return load_embedding_model(AtlasConfig.embedding_model)


@st.cache_resource
def ranking_model():
    return load_reranking_model(AtlasConfig.reranking_model)


def new_reader(documents):
    config = AtlasConfig(generation_model=os.getenv("ATLAS_GENERATION_MODEL")
                         or AtlasConfig.generation_model)
    reader = Atlas(
        config, retriever=Retriever(model=embedding_model()),
        reranker=Reranker(model=ranking_model()),
        generator=Generator(config.generation_model, max_output_tokens=config.max_output_tokens)
    )
    # Indexes and uploaded text are session-owned, never globally cached.
    reader.index_documents(documents)
    return reader


def json_download(label, value, filename, key):
    st.download_button(label, json.dumps(value, indent=2, ensure_ascii=False),
                       filename, "application/json", key=key)


def show_result(result, *, key):
    generation = result.get("generation")
    if generation:
        st.text(generation["answer"])
        if generation["status"] != "answered":
            st.info(generation.get("reason") or generation["status"].replace("_", " "))
    else:
        st.info("Evidence preview only. Generate an answer when you want synthesis.")
    with st.expander("Inspect source passages and supporting quotes"):
        if generation:
            for claim in generation.get("claims", []):
                st.text(claim["text"])
                for evidence in claim["evidence"]:
                    st.text(f'[{evidence["passage_id"]}] "{evidence["quote"]}"')
        for passage in result["context"]:
            st.caption(f'[{passage["id"]}] {passage["source"]} · characters {passage["start"]}:{passage["end"]}')
            st.text(passage["text"])
    json_download("Export this answer and its evidence", result, "atlas-answer.json", key)


def discover_page():
    st.header("Find papers worth reading")
    st.write("Enter a research topic. Compare relevant papers using publication context, "
             "scholarly uptake, traceable affiliations, and known editorial updates.")
    with st.form("discovery_form"):
        topic = st.text_input("Research topic", placeholder="e.g. retrieval-augmented generation evaluation")
        c1, c2, c3 = st.columns(3)
        preference = c1.selectbox("Reading goal", ["Balanced", "Foundational", "Recent"])
        count = c2.slider("Papers to shortlist", 3, 10, 5)
        oa_only = c3.checkbox("Open-access copies only")
        since = st.text_input("Published from year (optional)", placeholder="2021")
        submitted = st.form_submit_button("Find papers")
    if submitted:
        st.session_state.pop("discovery_report", None)
        try:
            year = int(since) if since.strip() else None
            metadata = MetadataClient(
                cache_dir=ROOT / ".cache/metadata",
                openalex_key=os.getenv("OPENALEX_API_KEY") or None,
                contact_email=os.getenv("CROSSREF_CONTACT_EMAIL") or None
            )
            try:
                with st.spinner("Searching and checking bibliographic evidence…"):
                    report = PaperDiscovery(metadata, Reranker(model=ranking_model())).recommend(
                        topic, count=count, preference=preference.lower(), since_year=year,
                        open_access=oa_only
                    )
            finally:
                metadata.close()
            st.session_state["discovery_report"] = report
        except (MetadataError, ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
    report = st.session_state.get("discovery_report")
    if not report:
        return
    st.caption(f'{len(report["recommendations"])} recommendations from '
               f'{report["candidate_count"]} fetched records; '
               f'{report["checked_pool_count"]} top-relevance candidates checked for editorial updates.')
    st.info("These are reading recommendations, not certifications of research quality. "
            "A citation may support or criticize a paper; affiliation prestige does not establish accuracy.")
    for warning in report["warnings"]:
        st.warning(warning)
    for position, item in enumerate(report["recommendations"], 1):
        paper = Paper(**item["paper"])
        with st.container(border=True):
            st.subheader(f"Reading {position}")
            st.text(paper.title)
            citations = f"{paper.citation_count:,} citing works" if paper.citation_count is not None else "Citation count unavailable"
            st.caption(f"{paper.year or 'Year unavailable'} · {paper.provider} · {citations}")
            for i, reason in enumerate(item["reasons"], 1):
                st.write(f'{i}. {reason["label"]}')
                st.text(reason["text"])
                if safe_url(reason.get("source_url")):
                    st.link_button("Verify this reason", reason["source_url"], key=f"reason_{position}_{i}")
            if paper.integrity_status in {"correction_or_update", "check_failed", "partial_check"}:
                st.warning(paper.integrity_status.replace("_", " ").capitalize())
            st.caption(f"Metadata retrieved: {paper.fetched_at}")
            st.caption(f"Editorial check: {paper.integrity_checked_at or 'Unavailable'}")
            columns = st.columns(3)
            landing = safe_url(paper.landing_url or (f"https://doi.org/{paper.doi}" if paper.doi else paper.id))
            if landing:
                columns[0].link_button("Open publication", landing)
            if safe_url(paper.oa_url):
                columns[1].link_button("Open available copy", paper.oa_url)
            if columns[2].button("Select for reading", key=f"select_{position}"):
                clear_reader()
                st.session_state.pop("paper_upload", None)
                st.session_state.pop("paper_source", None)
                st.session_state["selected_paper"] = paper.to_dict()
                st.session_state["navigate_to"] = "Read & ask"
                st.rerun()
            with st.expander("Abstract, citing papers, and caveats"):
                st.text(paper.abstract or "No abstract available.")
                for cited in paper.citing_examples:
                    st.text(f'{cited.get("title")} ({cited.get("year") or "year unavailable"})')
                    if safe_url(cited.get("url")):
                        st.link_button("Open citing paper", cited["url"], key=f"citing_{position}_{cited['url']}")
                for caveat in item["caveats"]:
                    st.text("• " + caveat)
                if paper.integrity_url:
                    st.link_button("Inspect editorial metadata", paper.integrity_url)
                for notice in paper.notices:
                    st.text(notice["title"])
                    if safe_url(notice.get("url")):
                        st.link_button("Read editorial notice", notice["url"], key=f"notice_{position}_{notice['url']}")
            st.code(paper.reference, language=None)
            st.download_button("Export citation (RIS)", reference_ris(paper),
                               f"atlas-reading-{position}.ris", "application/x-research-info-systems",
                               key=f"ris_{position}")
    if not report["recommendations"]:
        st.info("No eligible papers remained. Try a more specific topic or broader date filter.")
    json_download("Export reading list with provenance", report, "atlas-reading-list.json", "reading_export")


def clear_reader():
    for key in ["reader", "active_upload", "overview", "history", "last_result", "evaluation_report"]:
        st.session_state.pop(key, None)


def read_page():
    st.header("Read a paper and ask evidence-backed questions")
    selected = st.session_state.get("selected_paper")
    if selected:
        st.caption("Selected recommendation")
        st.text(selected["title"])
        st.caption("Upload the corresponding PDF or text. Atlas does not assume that a filename proves identity.")
    mode = st.radio("Paper source", ["Upload my paper", "Try the synthetic example"], horizontal=True, key="paper_source")
    uploaded = None
    if mode == "Upload my paper":
        uploaded = st.file_uploader("PDF, text, or Markdown (25 MB maximum)", type=["pdf", "txt", "md"],
                                    max_upload_size=25, key="paper_upload")
        payload = uploaded.getvalue() if uploaded else None
        name = uploaded.name if uploaded else None
    else:
        path = ROOT / "data/examples/sleep_study.txt"
        payload, name = path.read_bytes(), path.name
        st.caption("Synthetic sleep-study fixture; this is not a published research paper.")
    fingerprint = hashlib.sha256(payload).hexdigest() if payload else None
    active = st.session_state.get("active_upload")
    # An empty uploader after page navigation is not a new selection.
    if active and (mode != active.get("mode") or (fingerprint and fingerprint != active["fingerprint"])):
        clear_reader()
        st.info("Paper selection changed. Load the new paper to start a fresh reading session.")
    if st.button("Load paper", disabled=payload is None):
        try:
            with st.spinner("Extracting text and indexing source passages…"):
                parsed = parse_upload(name, payload)
                reader = new_reader(parsed.documents)
            clear_reader()
            st.session_state["reader"] = reader
            st.session_state["active_upload"] = {
                "filename": parsed.filename, "fingerprint": parsed.fingerprint, "mode": mode,
                "warnings": parsed.warnings, "document_count": len(parsed.documents)
            }
            st.session_state["history"] = []
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
    reader = st.session_state.get("reader")
    if reader is None:
        st.info("Load one paper to enable summaries, evidence previews, and Q&A.")
        return
    active = st.session_state["active_upload"]
    st.success(f'Loaded {active["filename"]} · {active["document_count"]} text page(s)/document(s) · '
               f'{len(reader.retriever.chunks)} indexed passages')
    for warning in active["warnings"]:
        st.warning(warning)
    confirmed = st.checkbox("I confirm this upload corresponds to the selected recommendation",
                            key="confirm_" + active["fingerprint"] + (selected or {}).get("id", ""),
                            disabled=not selected)
    if selected and confirmed:
        st.caption("Recommendation association: user-confirmed; document hash retained in the export.")
    tabs = st.tabs(["Overview", "Ask questions", "Review retrieval"])
    enabled = bool(os.getenv("OPENAI_API_KEY"))
    with tabs[0]:
        st.write("Get a cited overview of the question, approach, findings, and limitations.")
        if not enabled:
            st.info("Add OPENAI_API_KEY to .env and restart the app to enable generation.")
        if st.button("Generate paper overview", disabled=not enabled):
            try:
                with st.spinner("Building four evidence-backed sections…"):
                    st.session_state["overview"] = build_overview(reader)
            except (GenerationError, ValueError, RuntimeError) as error:
                st.error(str(error))
        overview = st.session_state.get("overview")
        if overview:
            st.caption(overview["coverage_note"])
            st.caption(overview["citation_note"])
            for i, section in enumerate(overview["sections"]):
                st.subheader(section["heading"])
                show_result(section["result"], key=f"overview_answer_{i}")
            exported = {**overview, "upload": active,
                        "selected_paper": selected if confirmed else None,
                        "association": "user_confirmed" if confirmed else "unverified"}
            json_download("Export complete paper overview", exported, "atlas-overview.json", "overview_export")
    with tabs[1]:
        st.caption("Suggested questions")
        for question in suggested_questions():
            st.text("• " + question)
        with st.form("question_form"):
            question = st.text_area("Your question", placeholder="What limitations does this paper explicitly report?")
            action = st.radio("Response", ["Generate cited answer", "Preview source evidence"], horizontal=True)
            ask = st.form_submit_button("Ask Atlas")
        if ask:
            if action == "Generate cited answer" and not enabled:
                st.error("Add OPENAI_API_KEY to .env to generate an answer.")
            else:
                try:
                    with st.spinner("Retrieving evidence…"):
                        result = reader.ask(question) if action == "Generate cited answer" else reader.search(question)
                    data = result.to_dict()
                    st.session_state["last_result"] = result
                    st.session_state["history"].append(data)
                except (GenerationError, ValueError, RuntimeError) as error:
                    st.error(str(error))
        for i, result in enumerate(st.session_state.get("history", [])):
            with st.expander(result["question"], expanded=i == len(st.session_state["history"]) - 1):
                show_result(result, key=f"history_export_{i}")
        if st.session_state.get("history"):
            json_download("Export this reading session", {
                "upload": active, "selected_paper": selected if confirmed else None,
                "association": "user_confirmed" if confirmed else "unverified",
                "answers": st.session_state["history"]
            }, "atlas-session.json", "session_export")
    with tabs[2]:
        st.write("Grade retrieved evidence against your question. These labels describe relevance, not paper credibility.")
        st.table([{"Match": label, "Meaning": definition} for label, definition in RUBRIC.items()])
        result = st.session_state.get("last_result")
        if not result:
            st.info("Ask a question or preview evidence to review the top three retrieved passages.")
        else:
            st.text(result.question)
            labels = {}
            query_key = hashlib.sha256(result.question.encode()).hexdigest()[:12]
            for rank, (chunk, score) in enumerate(result.reranked[:3], 1):
                identifier = chunk_id(chunk)
                with st.expander(f"Rank {rank} · {chunk.source}", expanded=True):
                    st.text(chunk.text)
                    match = st.selectbox("Relevance judgment",
                                         ["Unjudged", "Perfect", "Close", "Decent", "Bad"],
                                         key=f"grade_{active['fingerprint']}_{query_key}_{identifier}")
                    rationale = st.text_input("Why this grade?",
                                              key=f"why_{active['fingerprint']}_{query_key}_{identifier}")
                    if match != "Unjudged":
                        labels[identifier] = {
                            "grade": {"Perfect": 3, "Close": 2, "Decent": 1, "Bad": 0}[match],
                            "rationale": rationale
                        }
            diagnostics = graded_diagnostics(result.reranked, labels)
            def value(number):
                return "Unknown" if number is None else str(number)
            cols = st.columns(4)
            cols[0].metric("Hit@1 · Perfect evidence", value(diagnostics["hit_at_1"]))
            cols[1].metric("Hit@3 · Perfect evidence", value(diagnostics["hit_at_3"]))
            cols[2].metric("Best at rank 1", diagnostics["best_match_at_1"])
            cols[3].metric("Best in top 3", diagnostics["best_match_at_3"])
            st.caption("1 = at least one Perfect match; 0 = no Perfect match after review. "
                       "Unknown = incomplete judgments. NDCG uses the reviewed pool as its ideal ranking.")
            answerable = st.checkbox("This paper contains enough evidence to answer the question",
                                     value=True, key=f"answerable_{active['fingerprint']}_{query_key}")
            case = {"query": result.question, "answerable": answerable, "labels": labels,
                    "label_scope": "judged_pool"}
            json_download("Export relevance labels", [case], "atlas-reviewed-case.json", "labels_export")


def evaluate_page():
    st.header("Understand retrieval performance")
    st.write("Run a fully labeled synthetic demonstration or evaluate the paper in your reading session.")
    st.table([{"Match": label, "Meaning": definition} for label, definition in RUBRIC.items()])
    uploaded_labels = st.file_uploader("Optional reviewed-case JSON for the active paper", type=["json"], key="labels_upload")
    if st.button("Evaluate retrieval"):
        st.session_state.pop("evaluation_report", None)
        try:
            if uploaded_labels:
                reader = st.session_state.get("reader")
                if reader is None:
                    raise ValueError("Load your paper in Read & ask before evaluating its labels.")
                from src.atlas.evaluation import validate_cases
                cases = validate_cases(json.loads(uploaded_labels.getvalue()))
            else:
                reader = new_reader(load_documents(ROOT / "data/graded_demo"))
                cases = load_cases(ROOT / "data/evaluation/graded_cases.json")
            with st.spinner("Evaluating retrieval without generation API calls…"):
                report = evaluate(reader, cases)
                report["dataset"] = "User-reviewed active paper" if uploaded_labels else "Synthetic demonstration"
                st.session_state["evaluation_report"] = report
        except (ValueError, OSError, RuntimeError) as error:
            st.error(str(error))
    report = st.session_state.get("evaluation_report")
    if not report:
        return
    st.caption(report["dataset"] + ". Retrieval relevance metrics do not measure answer accuracy.")
    summary = report["summary"]
    for system in ["dense", "reranked"]:
        st.subheader(system.capitalize())
        metrics = summary[system]
        cols = st.columns(3)
        for column, key, label in zip(cols, ["hit_at_1", "hit_at_3", "ndcg_at_3"],
                                     ["Hit@1", "Hit@3", "NDCG@3"]):
            number = metrics[key]
            column.metric(label, "Unknown" if number is None else f"{number:.2f}")
            column.caption(f"{metrics['metric_case_counts'][key]} judged answerable case(s)")
        st.write("Best match found in the top three:", metrics["best_at_3_counts"])
    for case in report["cases"]:
        with st.expander(case["query"]):
            st.table([
                {"System": system, "Rank": item["rank"], "Match": item["match"],
                 "Source": item["source"], "Rationale": item["rationale"]}
                for system in ["dense", "reranked"]
                for item in case[system]["matches"]["top_3"]
            ])
    json_download("Export full evaluation", report, "atlas-evaluation.json", "evaluation_export")


st.title("Atlas")
st.caption("Discover what to read. Understand what it says. Inspect the evidence.")
# Streamlit drops widget state for widgets not drawn on a run; keep the paper source across pages.
if "paper_source" in st.session_state:
    st.session_state["paper_source"] = st.session_state["paper_source"]
if "navigate_to" in st.session_state:
    st.session_state["page"] = st.session_state.pop("navigate_to")
page = st.sidebar.radio("Workflow", ["Discover", "Read & ask", "Evaluate"], key="page")
st.sidebar.caption("Broad research discovery · Local reading sessions")
if page == "Discover":
    discover_page()
elif page == "Read & ask":
    read_page()
else:
    evaluate_page()
