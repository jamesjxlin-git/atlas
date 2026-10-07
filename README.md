# Atlas — Discover, understand, and verify research

Atlas is an evaluation-driven research assistant I am building for people who already know their field, but cannot realistically keep up with everything changing inside it.

The idea came from something I kept seeing while working in healthcare consulting. I spent time around physicians and other senior professionals who had spent years becoming experts in their areas, yet the research landscape was moving too quickly for any one person to continuously read, compare, and validate everything that might matter.

The problem was rarely whether they *could* understand a paper. The harder problem was deciding what deserved their attention in the first place, getting oriented to unfamiliar developments quickly, and knowing whether an AI-generated explanation actually matched what the research said.

That is the gap Atlas is trying to address.

Instead of treating research Q&A as one prompt to a language model, Atlas keeps the workflow explicit:

```text
research topic
    ↓
paper discovery
    ↓
transparent reading shortlist
    ↓
selected paper
    ↓
retrieval
    ↓
reranking
    ↓
evidence inspection
    ↓
optional cited generation
    ↓
evaluation
```

The broader product goal is not to replace expertise. It is to help someone with deep expertise spend less time sorting through what changed and more time applying what they already know.

---

## Why Atlas?

There are already good tools for finding papers, and there are increasingly good tools for summarizing them.

I do not think the missing product is simply another search bar attached to an LLM.

The problem I am more interested in is what happens when someone is already senior in a field but is dealing with an information-bandwidth problem.

A physician may understand a therapeutic area extremely well and still need to get up to speed on a new technique. A researcher may enter an adjacent topic and have no obvious reason to know which five papers out of hundreds are the best place to begin. A technical leader may need to understand a new area quickly enough to make a decision without pretending that a fluent summary is equivalent to reading the evidence.

In those situations, Atlas is trying to answer two separate questions:

**What should I read?**

and then:

**What does the evidence actually say?**

That distinction drives most of the product and engineering decisions in the project.

Atlas does not treat citation count as truth, does not assume a prestigious affiliation makes a paper correct, and does not treat a retrieved quote as proof that a generated claim is semantically supported.

Where the system cannot establish something, I would rather make that uncertainty visible than smooth it over.

---

# Current capabilities

Atlas currently supports three connected workflows:

1. **Discover** — find and compare scholarly papers using public metadata.
2. **Read & ask** — upload a paper, retrieve evidence, inspect passages, and optionally generate a cited answer.
3. **Evaluate** — manually grade retrieval quality and calculate ranking metrics.

These pieces are intentionally modular so that changes to one layer can be evaluated without rebuilding the entire system.

---

# 1. Research discovery

Atlas searches scholarly metadata through OpenAlex with a Crossref fallback.

A user can:

- enter a research topic,
- request a balanced, foundational, or recent reading list,
- filter by publication year,
- filter for open-access availability,
- inspect citation counts,
- view citing-paper examples when available,
- inspect author affiliations,
- see publication context,
- check for known editorial updates,
- open source records,
- export the reading list,
- and export RIS citations.

The goal is not to produce an exhaustive systematic review.

It is to give the user a short, defensible starting point and make the reasons behind that shortlist inspectable.

---

## How paper discovery works

OpenAlex supplies the initial topic-search pool, normally 30 records.

Atlas then:

1. normalizes DOI identifiers,
2. removes duplicate records,
3. removes ineligible work types,
4. ranks titles and available abstracts with the cross-encoder,
5. selects a bounded candidate pool,
6. checks leading candidates for editorial updates through Crossref,
7. and applies an inspectable recommendation heuristic.

The editorial-check pool is normally:

```text
top 12 candidates
```

or twice the requested recommendation count if that is larger.

The final reading list is normally:

```text
3–5 recommended papers
```

depending on the requested count.

---

## Recommendation preferences

Atlas currently supports three recommendation modes:

```text
Balanced
Foundational
Recent
```

The current weights are:

| Preference | Relevance | Citation uptake | Recency |
| --- | ---: | ---: | ---: |
| Balanced | 0.80 | 0.10 | 0.10 |
| Foundational | 0.80 | 0.20 | 0.00 |
| Recent | 0.80 | 0.00 | 0.20 |

Relevance is based on the paper's position in the cross-encoder-ranked candidate pool.

The current relevance term is:

```text
1 / sqrt(rank)
```

Recency is:

```text
1 / (1 + age / 3)
```

Citation uptake uses the recorded citation percentile where available and otherwise falls back to a capped logarithmic citation-count heuristic.

Missing age or uptake information receives a neutral fallback value rather than being converted into a fabricated zero.

These weights are transparent product defaults.

They are not empirically optimized probabilities that a paper is "credible."

---

## What a recommendation reason means

Atlas attempts to show three factual reasons for each recommended paper when the metadata supports them.

Possible signals include:

- topical relevance,
- publication context,
- scholarly citation uptake,
- citing-paper examples,
- traceable author affiliations,
- publication recency,
- DOI traceability,
- and editorial-update status.

These are **selection signals**, not guarantees of scientific correctness.

A paper can be highly cited and still be wrong.

A citing paper can criticize the paper it cites.

An affiliation provides provenance but does not prove rigor.

A journal record does not prove reproducibility.

When Atlas cannot support a positive recommendation reason from available metadata, it exposes the gap instead of inventing a reason.

---

# 2. Editorial and integrity checks

When a DOI is available, Atlas checks Crossref for incoming editorial updates associated with the original work.

Known:

- retractions,
- and expressions of concern

are excluded from recommendations.

Corrections remain visible so that the user can inspect them.

Atlas distinguishes between:

```text
known editorial concern
correction found
no matching notice found
missing DOI
incomplete check
failed check
```

That distinction matters.

A successful query that returns no notice means only:

> no matching notice was found in the metadata returned at that time

It does **not** mean that the paper is verified, correct, reproducible, or free of problems.

Atlas does not use Crossref as a "credibility oracle."

---

# 3. Selecting and uploading a paper

Discovery and paper reading are deliberately connected through a manual handoff.

Atlas does not bypass paywalls or scrape arbitrary PDF URLs.

The user:

1. selects a recommended paper,
2. opens a publication or available copy,
3. uploads the corresponding file,
4. and can explicitly confirm that the upload matches the selected recommendation.

Supported upload types:

```text
PDF
TXT
Markdown
```

Current limits:

```text
25 MB
200 PDF pages
2,000,000 extracted characters
```

Scanned PDFs that contain no extractable text require OCR, which Atlas does not currently include.

---

## Document identity

The filename alone is not treated as proof that an uploaded file is the recommended paper.

Atlas retains a SHA-256 fingerprint of the upload in the reading-session metadata.

The recommendation-to-upload relationship remains unverified until the user confirms it.

This is a small product decision, but it prevents the system from silently claiming provenance it does not actually know.

---

# 4. PDF and text ingestion

For PDFs, Atlas extracts text page by page.

Source identifiers preserve the physical PDF page:

```text
paper.pdf#page=1
paper.pdf#page=2
paper.pdf#page=3
...
```

Each chunk also retains:

```text
source
start character offset
end character offset
original text
```

These offsets are relative to the extracted text.

They are not visual coordinates on the PDF page.

Complex layouts, multi-column documents, equations, tables, and extraction artifacts can therefore still require human inspection.

---

# 5. Chunking

Chunking has gone through several iterations during the project.

That evolution matters because one of the goals of Atlas is to understand *why* a system works, not just to keep layering components until the output looks good.

---

## Original baseline: fixed character windows

The first version used fixed-size character chunks with overlap.

Conceptually:

```text
characters 0–100
characters 80–180
characters 160–260
...
```

This was useful as a baseline because it was:

- simple,
- deterministic,
- fast,
- and easy to measure.

The limitation was obvious.

Character boundaries know nothing about language.

A chunk could begin or end in the middle of a word or sentence.

---

## Early sentence-grouping experiment

The next version grouped detected sentences together until a character budget would be exceeded.

This produced more coherent evidence than the original fixed-character baseline and helped motivate the move toward language-aware boundaries.

The early six-query experiment showed stronger query-passage similarity under the sentence-grouping approach while Hit@1 and Hit@3 remained saturated.

Example similarity changes included:

```text
0.587 → 0.729
0.641 → 0.762
0.642 → 0.813
0.676 → 0.848
```

At the time:

| Metric | Character baseline | Sentence grouping |
| --- | ---: | ---: |
| Hit@1 | 1.00 | 1.00 |
| Hit@3 | 1.00 | 1.00 |

That experiment did **not** establish improved retrieval accuracy because the tiny benchmark was already saturated.

It did show that boundary choices changed the quality of the passages being retrieved.

---

## Current implementation: token-bounded chunking with sentence-end preference

The current production path is more precise than the early character-based implementation.

Atlas uses the embedding model's fast tokenizer and creates chunks capped at:

```text
180 embedding tokens
```

with:

```text
32-token overlap
```

The chunker looks for a nearby sentence ending in the latter portion of the available token window.

If an appropriate sentence ending is available, Atlas prefers that location rather than cutting exactly at the maximum token count.

This helps chunks **end** more naturally.

However, the overlap itself is still token-based.

The next chunk begins by backing up 32 tokens from the previous endpoint.

That means the next passage can still begin:

- inside a sentence,
- in the middle of a clause,
- or occasionally in visually awkward text produced by PDF extraction.

The first real-paper evaluation made this limitation much more obvious.

Several highly relevant retrieved passages were technically correct but began halfway through a sentence or contained more surrounding information than I would want a user to read.

For that reason, I no longer describe the current implementation as fully "sentence-aware chunking."

A more accurate description is:

> **token-bounded chunking with sentence-end preference**

---

## Planned chunking improvement

The next chunking iteration should build chunks from complete sentence units first.

Rather than backing up an arbitrary token count, overlap would carry one or more **whole preceding sentences** into the next chunk.

A future version should roughly follow:

```text
extract text
    ↓
normalize PDF line breaks / hyphenation
    ↓
detect sentence units
    ↓
group complete sentences within token budget
    ↓
carry complete previous sentence(s) for overlap
    ↓
fallback to token splitting only for unusually long sentences
```

The goal is not just prettier text.

A research assistant should return evidence that a person can comfortably inspect.

Retrieval relevance and passage readability both matter.

The existing synthetic and real-paper benchmarks make this change measurable: the revised chunker can be compared against the current implementation rather than accepted because it "looks better."

---

# 6. Dense retrieval

Atlas uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to encode document chunks and user queries.

Document chunks are embedded once when a paper is loaded.

The embeddings are normalized and stored as float32 vectors in the session-owned local index.

At query time:

```text
question
    ↓
query embedding
    ↓
cosine similarity against chunk matrix
    ↓
rank candidates
```

Because embeddings are normalized, a dot product can be used as cosine similarity.

The default candidate pool is:

```text
top 20 chunks
```

---

## Why dense retrieval?

Dense retrieval can identify semantically related evidence even when the question and the passage do not use exactly the same vocabulary.

It also allows document embeddings to be computed once and reused for many questions.

This is especially useful for the current product, where a user may ask several questions against the same uploaded paper.

---

## Why not BM25 only?

A lexical system such as BM25 can be extremely effective when exact terminology matters.

Atlas currently uses dense retrieval because natural-language research questions often paraphrase the source.

That does not mean lexical retrieval is ruled out.

A future experiment could compare:

```text
dense only
BM25 only
hybrid dense + lexical
```

using the same human-reviewed benchmark.

---

# 7. Cross-encoder reranking

Dense retrieval is efficient, but the query and passage are encoded separately.

The embedding model never jointly reads:

```text
(question, passage)
```

Atlas therefore performs a second ranking stage using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The cross-encoder scores the query and candidate passage together.

The retrieval path becomes:

```text
all paper chunks
      ↓
dense similarity search
      ↓
top 20 candidates
      ↓
cross-encoder scoring
      ↓
reranked evidence
```

---

## Why not use the cross-encoder against every chunk?

Cross-encoders are more expensive at query time.

With a very large corpus, evaluating every `(query, chunk)` pair would be inefficient.

Dense retrieval provides inexpensive candidate generation.

The cross-encoder then spends more computation only on those leading candidates.

This is a common two-stage ranking pattern and maps well to Atlas's current one-paper workflow.

---

# 8. Evidence packing

After reranking, Atlas builds an evidence context from the strongest original passages.

The current limits are:

```text
up to 4 distinct passages
3,500-token evidence JSON budget
```

Evidence packing retains application-owned source metadata.

Generated summaries are never treated as retrievable evidence.

Every new question returns to the original uploaded paper text.

This prevents one generated answer from quietly becoming evidence for another answer.

---

# 9. Evidence preview

One design choice I care about is that Atlas does not require an LLM to determine whether retrieval worked.

The user can choose:

```text
Preview source evidence
```

and inspect the retrieved passages directly.

This makes the retrieval system independently testable.

If Atlas never retrieved the correct evidence, a fluent generation model should not be allowed to hide that failure.

Retrieval quality and generation quality are therefore evaluated separately.

---

# 10. Optional structured generation

Atlas contains an optional evidence-grounded generation layer.

The default configured model is:

```text
gpt-4.1-mini-2025-04-14
```

A generation request contains:

- the user's question,
- selected evidence passages,
- source identifiers,
- and the structured response schema.

The generator is not given arbitrary hidden conversational context from previous questions.

Questions are independent.

The current generator allows:

```text
up to 6 claims
up to 1,600 output tokens
```

The generation model is configurable through:

```text
ATLAS_GENERATION_MODEL
```

provided the selected model supports the required structured-output interface and tokenizer assumptions.

---

## Generation is optional

The following workflows do **not** require an OpenAI API key:

- scholarly discovery,
- paper ingestion,
- dense retrieval,
- cross-encoder reranking,
- evidence preview,
- retrieval evaluation,
- synthetic benchmarks.

An OpenAI API key is only required for:

- generated cited answers,
- generated paper overviews.

API usage has separate billing.

---

# 11. Evidence and citation validation

Atlas validates the structure of generated evidence before showing a generated answer.

A generated claim must reference a valid retrieved passage identifier.

A quoted piece of evidence must appear in the corresponding original passage, allowing whitespace normalization.

Citation metadata is rendered from application-owned source information rather than text invented by the model.

Atlas can therefore catch failures such as:

```text
unknown passage ID
invented supporting quote
malformed evidence structure
incomplete structured response
```

Those failures are not silently converted into a normal answer.

---

## Generation states

Atlas currently distinguishes:

```text
answered
insufficient_evidence
invalid_evidence
refused
```

Transport failures and incomplete API responses raise a separate generation error.

This avoids mislabeling an API failure as "the paper does not contain enough evidence."

---

# 12. The most important generation limitation

One of the most useful failures I found while building Atlas was that:

> **citation integrity is not the same thing as semantic entailment**

Atlas can verify that:

- the passage exists,
- the citation ID is valid,
- and the quote actually appears in that passage.

That still does not prove that the quote logically supports the generated claim.

I added an adversarial automated test that demonstrates this.

A genuine quote can be attached to a contradictory claim and still pass quote-presence validation.

Because of that, Atlas is **not** described as "hallucination-free."

The current validation establishes source and quote integrity.

Semantic support remains a separate evaluation problem.

Future work should measure entailment explicitly rather than pretending that citation presence solves it.

---

# 13. Paper overview

Atlas can optionally generate four targeted overview sections:

```text
Research question
Approach
Main findings
Limitations
```

Each section performs its own evidence retrieval and generation step.

The overview is therefore an evidence-selected summary rather than an exhaustive summary of every page.

Generated overview text is not fed back into the retrieval index.

---

# 14. Human-readable retrieval evaluation

A similarity score is useful to a model developer.

It is not always useful to the person reading the evidence.

Atlas therefore adds a human-reviewed relevance layer.

Each retrieved passage can be labeled:

| Grade | Label | Meaning |
| ---: | --- | --- |
| 3 | **Perfect** | Direct evidence sufficient for the requested answer |
| 2 | **Close** | Useful evidence with an important gap or missing qualification |
| 1 | **Decent** | Related background that does not directly answer the question |
| 0 | **Bad** | Irrelevant, wrong-scope, or misleading evidence |

These labels apply to one passage for one specific question.

They do **not** represent the credibility of the paper itself.

---

## Human review, not score thresholds

The labels are assigned manually.

Atlas does not map raw cosine or cross-encoder scores directly into Perfect / Close / Decent / Bad.

The intended workflow is:

```text
Atlas retrieves
      ↓
human reviews
      ↓
evaluation code calculates metrics
```

Unreviewed evidence remains:

```text
Unjudged
```

rather than being treated as Bad.

---

# 15. Retrieval metrics

Atlas reports several ranking metrics.

## Hit@1

Did the first-ranked passage contain Perfect evidence?

## Hit@3

Did any of the top three passages contain Perfect evidence?

## Useful Hit@3

Did the top three contain at least Close evidence?

## Mean Reciprocal Rank

MRR rewards systems that place the first Perfect passage earlier.

```text
Perfect at rank 1 → 1.0
Perfect at rank 2 → 0.5
Perfect at rank 3 → 0.333...
```

## NDCG@3

NDCG uses the complete graded relevance ordering rather than reducing every passage to relevant / irrelevant.

The gain function is:

```text
2^grade - 1
```

with logarithmic rank discount.

For example:

```text
[Close, Perfect, Decent]
```

has:

```text
Hit@1 = 0
Hit@3 = 1
```

while NDCG also reflects the ordering of all three judgments.

---

# 16. Label scope

Atlas distinguishes between:

```text
judged_pool
complete_corpus
```

A `complete_corpus` declaration is accepted only when every current chunk has a label.

A `judged_pool` NDCG score is conditional on the reviewed pool.

Chunk identifiers include the source, offsets, and original text.

Changing chunking or source identity invalidates old labels.

This prevents reviewed judgments from silently being applied to different passages.

---

# 17. Validation strategy

I did not want the first published paper I uploaded to also be the first time Atlas's evaluation logic was tested.

The project therefore moved through several layers:

```text
simple synthetic examples
      ↓
controlled graded synthetic benchmark
      ↓
live scholarly metadata
      ↓
human-reviewed published-paper pilot
```

Each layer answers a different question.

Synthetic data provides controlled edge cases.

Live metadata tests the discovery path against real APIs.

The published-paper pilot tests the retrieval system on authentic research language.

---

# 18. Automated test suite

The current project passes:

```text
68 / 68 automated tests
```

Latest local run:

```text
68 passed in 3.74s
```

The exact runtime can vary because of local caching and operating-system conditions.

The important result is that all 68 tests pass.

The suite covers chunking, token budgets, source offsets, embedding behavior, dense retrieval, reranking, evidence packing, structured response parsing, citation and quote validation, refusal and insufficiency behavior, scholarly metadata normalization, editorial notices, API fallbacks, relevance grading, ranking metrics, and Streamlit workflows.

Tests use deterministic stubs where appropriate and make no paid generation requests.

They verify implementation behavior, not the quality of every real-model answer.

---

# 19. Controlled synthetic retrieval benchmark

Atlas includes synthetic fixtures for:

- basic retrieval,
- answerable questions,
- unanswerable questions,
- distractor passages,
- RAG behavior,
- relevance grading,
- citation behavior,
- and failure cases.

The fully graded retrieval fixture contains:

```text
5 synthetic documents
4 labeled questions
3 answerable questions
1 intentionally unanswerable question
```

The corpus was indexed and searched using the real MiniLM embedding and cross-encoder models.

## Synthetic results

| Metric | Dense retrieval | Reranked |
| --- | ---: | ---: |
| Hit@1 — Perfect evidence | 0.667 | 0.667 |
| Hit@3 — Perfect evidence | 1.000 | 1.000 |
| MRR | 0.778 | **0.833** |
| NDCG@3 | 0.936 | **0.953** |

The reranker improved MRR and NDCG@3 while preserving a Perfect result in the top three for every answerable synthetic question.

This is a controlled smoke benchmark, not a general accuracy claim.

---

# 20. Human-reviewed real-paper pilot

After the controlled evaluation was stable, Atlas was tested against a published paper:

> Samuelson et al. — *Exploring innovation landscapes: a national cross-sectional study of Swedish primary care from the viewpoint of primary care managers.*

The pilot contained:

```text
6 human-reviewed questions
5 answerable
1 intentionally unanswerable
3 reviewed top passages per question
```

Questions covered study design, respondent count, collaboration patterns, barriers to innovation, causal limitations, representativeness, selection bias, and one intentionally unsupported causal-effect question.

The final question deliberately asked for a causal effect the paper could not establish.

That makes answerability itself part of the evaluation.

## Real-paper reranked results

| Metric | Result |
| --- | ---: |
| Hit@1 — Perfect evidence | **0.600** |
| Hit@3 — Perfect evidence | **1.000** |
| Useful Hit@3 | **1.000** |
| MRR | **0.800** |
| NDCG@3 — judged pool | **0.938** |

Across the five answerable questions:

```text
5 / 5
```

retrieved a Perfect passage somewhere in the top three.

Three of five placed Perfect evidence at rank one.

## Real-paper timing

| Stage | Mean |
| --- | ---: |
| Dense retrieval | ~4.1 ms |
| Cross-encoder reranking | ~21.5 ms |
| Context construction | ~0.5 ms |
| Combined retrieval → context | **~26 ms** |

Paper indexing took approximately:

```text
214 ms
```

on the local development machine.

These are local measurements, not production latency guarantees.

## Why I do not compare dense vs. reranked metrics in this pilot

Human judgments were collected for the reranked top-three passages.

Many dense-only passages therefore remained Unjudged.

The dense-side denominators are incomplete, so those values are not a fair head-to-head comparison.

A proper comparison would require pooled judgments from both systems or complete-corpus labels.

## Scope

This is still a small evaluation:

```text
1 published paper
6 questions
5 answerable cases
```

It demonstrates that the evaluation workflow works on authentic research text.

It does not establish cross-domain research accuracy.

---

# 21. Live scholarly-discovery validation

The discovery pipeline has also been run against live public scholarly metadata.

Two topics were tested:

```text
retrieval augmented generation evaluation
urban heat island mitigation green roofs
```

For each topic:

```text
30 records fetched
12 leading candidates checked for incoming editorial updates
3 recommendations returned
```

Missing metadata or citing-paper examples were left missing rather than invented.

A successful Crossref check with no matching notice means only that no matching notice was found in the returned metadata at that time.

---

# 22. Validation summary

| Layer | Current evidence |
| --- | --- |
| Automated engineering checks | **68 passing tests** |
| Synthetic development fixtures | Retrieval, RAG, distractors, answerability, grading, evidence validation |
| Fully graded synthetic benchmark | **5 documents / 4 questions** |
| Synthetic reranked performance | **1.000 Hit@3 / 0.833 MRR / 0.953 NDCG@3** |
| Live scholarly discovery | **2 topics × 30 fetched records** |
| Published-paper pilot | **6 human-reviewed questions** |
| Real-paper reranked performance | **1.000 Hit@3 / 0.800 MRR / 0.938 NDCG@3** |

The point of this progression is to move from controlled behavior to increasingly realistic evidence while keeping the limits of each test visible.

---

# 23. Current architecture

```text
Research topic
      │
      ▼
OpenAlex / Crossref
      │
      ▼
Recommendation ranking + editorial checks
      │
      ▼
User selects a paper
      │
      ▼
PDF / TXT / MD
      │
      ▼
Extraction + token-bounded chunking
with sentence-end preference
      │
      ▼
MiniLM dense embeddings
      │
      ▼
Top 20 candidates
      │
      ▼
Cross-encoder reranking
      │
      ▼
Ranked original evidence
      │
      ├────────────► Human inspection + evaluation
      │
      ▼
Optional structured generation
      │
      ▼
Citation + quote validation
      │
      ▼
Evidence-linked answer
```

---

# 24. Why exact in-memory vector search?

Atlas currently indexes one paper at a time.

At that scale, a normalized in-memory embedding matrix provides exact cosine ranking, low local latency, and simple inspectable behavior without an external vector-database dependency.

A vector database would make more sense for persistent multi-paper libraries, many simultaneous users, very large corpora, or distributed hosted retrieval.

---

# 25. Why no agent framework?

The current workflow is deterministic enough that explicit modules are easier to inspect, test, evaluate, and debug.

Atlas separates discovery, ingestion, chunking, retrieval, reranking, context, generation, and evaluation.

If future multi-step research behavior genuinely requires dynamic planning, that can be added later.

I do not want "agentic" complexity to become a feature by itself.

---

# 26. Why separate retrieval quality from generation quality?

A poor answer can come from different failures:

```text
wrong paper selected
retrieval missed the evidence
reranking ordered evidence poorly
context packing omitted the right passage
generation misinterpreted correct evidence
citation validation failed
```

Scoring only the final answer makes those failures difficult to diagnose.

Atlas evaluates retrieval independently before generation.

---

# 27. Streamlit application

The current web interface uses Streamlit.

Main pages:

```text
Discover
Read & ask
Evaluate
```

The interface supports scholarly search, reading-list inspection, citation export, paper upload, evidence preview, optional generation, manual relevance grading, reviewed-label export, and retrieval evaluation.

---

# 28. Running Atlas

Use Python 3.11 or 3.12.

```bash
git clone <YOUR-REPOSITORY-URL>
cd Atlas

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements-dev.txt

cp -n .env.example .env

python -m streamlit run app.py
```

Windows:

```bash
.venv\Scripts\activate
```

The first retrieval run may download the two MiniLM model weights.

---

# 29. Environment variables

```text
OPENAI_API_KEY=
ATLAS_GENERATION_MODEL=gpt-4.1-mini-2025-04-14

OPENALEX_API_KEY=
CROSSREF_CONTACT_EMAIL=
```

`OPENAI_API_KEY` is optional and only needed for generation.

`OPENALEX_API_KEY` is optional.

`CROSSREF_CONTACT_EMAIL` is optional.

Actual credentials belong in `.env`, which is excluded from Git.

---

# 30. CLI examples

## Discover papers

```bash
python -m src.atlas.cli discover   "urban heat island mitigation"   --count 5   --preference recent   --output reports/readings.json
```

## Preview local evidence

```bash
python -m src.atlas.cli ask   "What limitations are reported?"   --documents data/examples/sleep_study.txt   --retrieve-only
```

## Generate a cited answer

```bash
python -m src.atlas.cli ask   "What limitations are reported?"   --documents data/private/paper.pdf   --output reports/answer.json
```

## Generate an overview

```bash
python -m src.atlas.cli overview   --documents data/private/paper.pdf   --output reports/overview.json
```

## Run the synthetic graded evaluation

```bash
python -m src.atlas.cli evaluate   --output reports/graded.json
```

## Evaluate human-reviewed labels

```bash
python -m src.atlas.cli evaluate   --documents data/private/paper.pdf   --cases data/evaluation/my-reviewed-cases.json   --output reports/my-paper.json
```

Generation during evaluation is opt-in with:

```text
--generate
```

---

# 31. Example reviewed-evaluation workflow

```text
1. Load a paper
2. Ask or preview one question
3. Review the reranked top three passages
4. Assign Perfect / Close / Decent / Bad
5. Mark whether the paper contains enough evidence to answer
6. Export the reviewed case
7. Repeat
8. Combine reviewed cases
9. Run evaluation against the same paper/chunk configuration
```

Because chunk IDs depend on source identity, offsets, and original text, labels must be reused only with the same document and chunk configuration.

---

# 32. Repository structure

```text
Atlas/
│
├── app.py
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── examples/
│   ├── graded_demo/
│   ├── evaluation/
│   └── private/              # ignored
│
├── docs/
│   └── VALIDATION.md
│
├── src/
│   └── atlas/
│       ├── __init__.py
│       ├── chunking.py
│       ├── cli.py
│       ├── config.py
│       ├── context.py
│       ├── demo.py
│       ├── discovery.py
│       ├── evaluation.py
│       ├── generation.py
│       ├── grades.py
│       ├── ingestion.py
│       ├── metadata.py
│       ├── models.py
│       ├── overview.py
│       ├── pipeline.py
│       ├── reranking.py
│       ├── retrieval.py
│       └── scholar.py
│
└── tests/
    ├── test_app.py
    ├── test_chunking.py
    ├── test_discovery.py
    ├── test_grades.py
    └── test_rag.py
```

---

# 33. Module responsibilities

| Layer | Main files |
| --- | --- |
| Streamlit interface | `app.py`, `.streamlit/config.toml` |
| Scholarly providers | `scholar.py` |
| Metadata normalization | `metadata.py` |
| Recommendation logic | `discovery.py` |
| Upload parsing | `ingestion.py` |
| Chunk construction | `chunking.py` |
| Dense retrieval | `retrieval.py` |
| Cross-encoder ranking | `reranking.py` |
| Evidence packing | `context.py` |
| Structured generation | `generation.py` |
| Pipeline orchestration | `pipeline.py` |
| Paper overview | `overview.py` |
| Graded relevance | `grades.py` |
| Retrieval evaluation | `evaluation.py` |
| Model loading | `models.py` |
| Configuration | `config.py` |
| CLI | `cli.py` |

---

# 34. Data handling

Uploaded paper text and the local retrieval index remain associated with the active reading workflow.

Private documents should remain under:

```text
data/private/
```

and are excluded from Git.

The following should also remain excluded:

```text
.env
Streamlit secrets
private papers
local generated reports where configured
```

When optional hosted generation is used, selected evidence and the question are sent to the configured model provider.

Atlas is a research-assistance prototype and is not a clinical decision-support system.

---

# 35. Security and trust boundaries

Atlas treats uploaded paper text and user questions as untrusted prompt data.

Prompt boundaries reduce accidental instruction mixing but do not constitute a complete prompt-injection defense.

The project does not claim complete protection against adversarial document instructions, indirect prompt injection, semantic citation misuse, or model-level jailbreak behavior.

Those need separate evaluation before production deployment.

---

# 36. Current limitations

Atlas remains a research prototype.

### Benchmark size

The real-paper benchmark currently covers one published paper and six reviewed questions.

### Chunk readability

The current token overlap can cause chunks to begin partway through a sentence.

### PDF extraction

Line breaks, hyphenation, tables, equations, and multi-column layouts may be imperfect.

### OCR

Image-only PDFs are not supported.

### Semantic entailment

Quote validation does not prove that a quote supports the associated claim.

### Scholarly metadata

OpenAlex and Crossref coverage is incomplete and changes over time.

### Recommendation quality

The recommendation weights are transparent heuristics rather than empirically calibrated probabilities of paper quality.

### Generation evaluation

The structured generation layer is implemented and covered by automated tests using mocked HTTP responses, but a larger live benchmark of answer faithfulness, abstention, prompt-injection resilience, and multilingual performance remains future work.

---

# 37. Development roadmap

## Completed — retrieval foundation

- [x] character baseline
- [x] sentence grouping experiment
- [x] tokenizer-aware chunking
- [x] dense embeddings
- [x] exact cosine retrieval
- [x] top-k candidate retrieval
- [x] cross-encoder reranking
- [x] synthetic evaluation
- [x] Hit@1 / Hit@3
- [x] MRR
- [x] graded relevance
- [x] NDCG@3
- [x] answerability labels
- [x] unanswerable cases

## Completed — discovery

- [x] OpenAlex search
- [x] Crossref fallback
- [x] DOI normalization
- [x] deduplication
- [x] recommendation preferences
- [x] citation metadata
- [x] affiliations
- [x] citing-paper examples
- [x] editorial-update checks
- [x] retraction / concern handling
- [x] reading-list export
- [x] RIS export

## Completed — reading workflow

- [x] PDF / TXT / Markdown upload
- [x] page-aware PDF extraction
- [x] document fingerprints
- [x] evidence preview
- [x] structured generation layer
- [x] source-ID validation
- [x] exact quote validation
- [x] insufficiency / invalid / refusal states
- [x] four-section overview
- [x] Streamlit workflow

## Completed — evaluation

- [x] four-level manual relevance grading
- [x] reviewed-label exports
- [x] judged-pool evaluation
- [x] complete-corpus safeguards
- [x] synthetic benchmark
- [x] published-paper pilot
- [x] 68 automated tests

## Next — chunk readability

- [ ] normalize PDF line-break artifacts
- [ ] normalize split-word hyphenation where safe
- [ ] construct chunks from complete sentence units
- [ ] use whole-sentence overlap
- [ ] retain hard token ceilings
- [ ] handle unusually long individual sentences
- [ ] rerun the current synthetic and real-paper benchmarks
- [ ] evaluate readability separately from retrieval relevance

## Next — larger evaluation

- [ ] multiple published papers
- [ ] 25–60+ reviewed questions
- [ ] multiple research domains
- [ ] independent reviewers
- [ ] inter-rater agreement
- [ ] pooled dense + reranked judgments
- [ ] held-out evaluation questions

## Next — semantic support

- [ ] entailment-oriented checks
- [ ] contradiction detection
- [ ] claim-to-source support metrics
- [ ] adversarial faithfulness cases

## Future

Potential directions include:

- BM25 / hybrid retrieval,
- alternative embeddings and rerankers,
- multi-paper retrieval,
- persistent research libraries,
- conflicting-source comparison,
- query expansion,
- semantic entailment evaluation,
- local generation backends,
- and user studies with domain experts.

---

# 38. Design principles

**Measure before assuming improvement.**

New components should be evaluated against a baseline.

**Prefer inspectable systems.**

Users should be able to see why a paper was recommended and what evidence an answer used.

**Keep components modular.**

Retrieval, reranking, generation, and evaluation should be independently replaceable.

**Separate retrieval from generation.**

A retrieval failure and a generation failure are different problems.

**Do not turn metadata into truth.**

Citations and affiliations are signals, not guarantees.

**Do not turn citations into entailment.**

A real source can still be interpreted incorrectly.

**Document neutral and negative results.**

A change that does not improve a metric can still teach something.

**Increase complexity only when justified.**

Vector databases, agents, hybrid search, and model-based judges are experiments to evaluate, not automatic upgrades.

---

# 39. What Atlas is not

Atlas is not:

- a systematic-review engine,
- a guarantee that recommended papers are correct,
- a clinical decision-support system,
- a replacement for domain expertise,
- or a claim of hallucination-free generation.

It is an evaluation-driven research-assistance prototype built to make the evidence path more visible.

---

# 40. Where I want to take Atlas next

The first version answered an engineering question for me:

> Can I build a transparent retrieval and evaluation pipeline that works beyond a toy example?

The next stage is more product-driven.

The chunking issue from the real-paper pilot is a good example. Atlas retrieved the right information, but some passages began halfway through sentences or included more surrounding material than a user should have to parse.

Technically correct retrieval is not enough if the evidence is unpleasant to read.

The next chunking iteration will therefore focus on complete sentence units, whole-sentence overlap, and PDF text cleanup. The existing benchmark gives me a way to test whether readability improves without degrading retrieval.

Beyond that, I want to expand the evaluation across more papers and domains, use independent reviewers, measure inter-rater agreement, and test semantic entailment separately from quote integrity.

The long-term product is not meant to replace someone who has spent twenty years becoming an expert in a field.

It should help that person spend less of their limited time sorting through what changed and more of it applying the expertise they already have.

---

# 41. Primary references

- [OpenAlex API](https://help.openalex.org/api/)
- [OpenAlex work metadata](https://help.openalex.org/data/works/attributes/)
- [Crossref REST API](https://github.com/CrossRef/rest-api-doc)
- [Crossref Retraction Watch metadata](https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Streamlit](https://docs.streamlit.io/)
- [Sentence Transformers — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Cross Encoder — ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

---

# 42. Current status

```text
68 automated tests passing

Controlled synthetic retrieval benchmark
5 documents
4 questions
Hit@3:   1.000
MRR:     0.833
NDCG@3:  0.953

Human-reviewed published-paper pilot
6 questions
5 answerable
1 intentionally unanswerable
Hit@1:   0.600
Hit@3:   1.000
MRR:     0.800
NDCG@3:  0.938

Live scholarly discovery
2 topics
30 fetched records per topic
12 leading candidates checked
3 recommendations returned per run
```

Atlas is still being built.

The objective is not to make research uncertainty disappear behind a polished answer. It is to help someone who already has expertise keep pace with a changing field while making the paper-selection logic, retrieved evidence, evaluation results, and system limitations easier to inspect.
