"""Targeted paper overviews reuse the same evidence-backed RAG pipeline."""

OVERVIEW_QUESTIONS = {
    "Research question": "What problem or research question does this paper investigate, and why?",
    "Approach": "What methods, data, experiments, or study design does the paper report?",
    "Main findings": "What are the main findings or contributions reported in this paper?",
    "Limitations": "What limitations, uncertainty, or unresolved questions does the paper explicitly report?"
}


def build_overview(atlas):
    sections = []
    for heading, question in OVERVIEW_QUESTIONS.items():
        result = atlas.ask(question)
        sections.append({"heading": heading, "result": result.to_dict()})
    return {
        "sections": sections,
        "coverage_note": "Each section uses targeted retrieval from original paper text. "
                         "This is an evidence-selected overview, not an exhaustive review of every page.",
        "citation_note": "Passage numbers restart within each section.",
        "tokens": {
            key: sum(section["result"]["generation"]["usage"][key] for section in sections)
            for key in ("input_tokens", "output_tokens")
        }
    }


def suggested_questions():
    return [
        "What knowledge gap does this paper address?",
        "How were the methods or experiments designed?",
        "What findings support the main conclusion?",
        "Which limitations are explicitly stated?",
        "Which claims would need additional evidence?",
        "Explain the paper's main contribution in plain language."
    ]
