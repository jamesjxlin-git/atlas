# Validation record

Validated on October 6, 2026 with Python 3.12. No remote repository was changed or deployed.

## Automated checks

67 tests passed, including:

- Token budgets and original-text offsets; embedding caching and invalid-vector handling.
- Structured response parsing through the real OpenAI SDK with mocked HTTP responses.
- Citation IDs, supporting quote presence, unsupported evidence, refusals, incomplete responses, and sanitized API failures.
- Four-section overview orchestration against original evidence, including request count and token aggregation.
- Provider metadata normalization, missing versus zero citations, DOI deduplication, editorial notices targeting the original DOI, and exclusion of retractions/concerns.
- Bounded HTTP retries, fallback behavior, timestamp-preserving metadata caches, avoiding credential serialization, and actionable model-loading failures.
- Perfect/Close/Decent/Bad judgments, unknown-label handling, graded NDCG, and full-corpus label coverage.
- Streamlit application tests for navigation, discovery-card selection, loading a synthetic paper, retrieval previews, manual grading, evaluation display, and clearing prior reading state on paper changes.

Tests inject local tokenizers and inference stubs. They make no paid requests and do not download model weights. They verify implementation behavior, not the quality of real-model answers.

## Actual local retrieval

The graded synthetic corpus was indexed and searched using the real MiniLM embedding and cross-encoder models. Three answerable questions and one unanswerable question were evaluated. The author supplied fixture grades; there was no independent physician/researcher review.

| Metric | Dense retrieval | Reranked |
| --- | --- | --- |
| Hit@1, Perfect evidence | 0.667 | 0.667 |
| Hit@3, Perfect evidence | 1.000 | 1.000 |
| MRR within candidates | 0.778 | 0.833 |
| NDCG@3, complete labeled corpus | 0.936 | 0.953 |

All averages above use three answerable cases. This tiny synthetic example is a smoke demonstration. It cannot establish a significant retrieval improvement or predict performance on real papers. Do not present these numbers as benchmark accuracy.

The causal-explanation question returned Close evidence at rank 1 and Perfect evidence at rank 2 after reranking. Its Hit@1 was 0 and Hit@3 was 1. The named-university question had Bad evidence in the top three and was excluded from answerable retrieval averages.

## Live public metadata

The topic “retrieval augmented generation evaluation” completed a live discovery run: 30 fetched records, 12 candidates checked for incoming editorial updates, and three recommendations exported with three supported reasons each. Returned citing-paper examples included zero, one, and three examples across the shortlisted records. Missing examples were not invented.

The successful checks reported no matching notice in the queried Crossref results at their recorded check times. That is an observation about available metadata, not proof of research integrity. Retraction, concern, correction, incomplete-check, and failure behavior was tested with controlled HTTP fixtures.

A second live topic, “urban heat island mitigation green roofs,” also completed with 30 fetched records, 12 editorial checks, and three recommendations. This used already downloaded model weights in offline mode after a model-refresh request encountered a network restriction. First-run model downloading still needs access to the model host; scholarly discovery continues to need public API access.

## Paid generation scope

No live paid OpenAI generation was run because no generation API key was supplied. The real SDK's structured parsing and the generation pipeline were tested with HTTP mocks. Real-paper summaries, semantic entailment, insufficiency decisions, prompt-injection resilience, and multilingual/domain-specific quality still need held-out evaluation.

Quote validation deliberately does not establish entailment. A test demonstrates that a real supporting quote can be attached to a contradictory claim and still pass quote-presence validation. This is disclosed rather than labeled “hallucination-free.”

## Tested dependency versions

| Dependency | Version |
| --- | --- |
| sentence-transformers | 5.7.0 |
| transformers | 5.18.0 |
| torch | 2.14.1 |
| openai | 2.54.0 |
| streamlit | 1.65.0 |
| pypdf | 6.10.0 |
| tiktoken | 0.14.0 |
| pytest | 9.1.1 |

Requirements specify compatible major-version ranges. Re-run the tests when resolving a different environment. The original DistilBART summary experiment was retained but not revalidated with a live model run.
