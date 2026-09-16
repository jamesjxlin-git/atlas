# Atlas

Atlas is an evaluation-driven AI research assistant designed to retrieve, rank, and eventually synthesize evidence from source documents.

The project is being built from the retrieval layer upward rather than beginning with a high-level RAG framework. The goal is not only to build a working AI assistant, but to understand and measure how individual system-design choices affect retrieval and answer quality.

Atlas currently implements sentence-aware document chunking, dense vector retrieval, cross-encoder reranking, and retrieval evaluation. Future stages will add grounded generation, citations, answer evaluation, and multi-step research capabilities.

## Why Atlas?

A research assistant needs to do more than send a prompt to a language model. It needs to determine what information is relevant, retrieve appropriate evidence, distinguish strong evidence from loosely related passages, generate an answer grounded in those sources, and evaluate whether the result is actually supported.

Atlas is therefore being developed as a modular pipeline:

```text
Documents
    ↓
Parsing & Chunking
    ↓
Embedding
    ↓
Dense Retrieval
    ↓
Candidate Passages
    ↓
Cross-Encoder Reranking
    ↓
Ranked Evidence
    ↓
Grounded Generation          [planned]
    ↓
Citations                    [planned]
    ↓
Answer Evaluation            [planned]
    ↓
Feedback / Adaptation        [future]
```

Each stage is kept relatively independent so that alternative approaches can be tested without rebuilding the entire system.

---

## Current Architecture

### 1. Document Chunking

Documents must be divided into smaller units before they can be embedded and retrieved.

Atlas originally used fixed-size character chunks with overlap.

```text
Document
   ↓
Characters 0–100
Characters 80–180
Characters 160–260
...
```

This approach was chosen as the initial baseline because it is simple, deterministic, inexpensive, and easy to evaluate.

However, it introduced an important problem: character boundaries have no understanding of language. Words and sentences could be split between chunks, producing passages such as:

```text
"nformation relevant to a query..."
```

or:

```text
"ing external sources."
```

### Current approach: sentence-aware chunking

Atlas now identifies sentence boundaries and groups complete sentences together until the target chunk size would be exceeded.

```text
Document
    ↓
Sentence 1
Sentence 2
Sentence 3
...
    ↓
Group sentences within target size
    ↓
Semantic chunks
```

This preserves more coherent evidence for retrieval.

The implementation also retains:

- source information
- starting character position
- ending character position

These metadata will later support source attribution and citations.

### Alternatives considered

**Character-based chunking**

Advantages:
- simple
- fast
- deterministic
- easy to overlap

Disadvantages:
- can split words and sentences
- chunk boundaries do not correspond to semantic boundaries

**Word-based chunking**

Advantages:
- avoids splitting individual words
- remains simple

Disadvantages:
- can still split sentences and ideas
- word count does not correspond directly to model token count

**Token-based chunking**

Advantages:
- aligns chunk size with model context limits
- useful when managing embedding or generation token budgets

Disadvantages:
- token boundaries still do not necessarily represent semantic boundaries
- requires tokenizer-specific logic

**Sentence-aware chunking — current choice**

Advantages:
- preserves natural language boundaries
- produces more coherent evidence
- remains relatively lightweight

Disadvantages:
- sentence lengths vary
- simple regular-expression sentence detection can mishandle abbreviations and unusual punctuation
- does not guarantee optimal semantic boundaries

**Semantic chunking — possible future experiment**

A more advanced approach could use embeddings or topic changes to identify when the meaning of a document shifts.

This may produce better semantic units, but introduces additional computation and complexity. Atlas will only adopt this approach if evaluation demonstrates that it improves retrieval.

---

## Chunking Experiment

The first Atlas experiment compared fixed-size character chunking against sentence-aware chunking.

The same embedding model, queries, and retrieval procedure were used so that chunking strategy was the primary changed variable.

Sentence-aware chunking increased query-passage similarity for all six initial evaluation queries.

Examples of top-result similarity changes included:

```text
0.587 → 0.729
0.641 → 0.762
0.642 → 0.813
0.676 → 0.848
```

Retrieval accuracy remained:

| Metric | Character Chunking | Sentence-Aware Chunking |
|---|---:|---:|
| Hit@1 | 1.00 | 1.00 |
| Hit@3 | 1.00 | 1.00 |

This does **not** establish that sentence-aware chunking improves retrieval accuracy. The initial six-query benchmark is small, and the character baseline already retrieves the expected evidence for every query.

The experiment instead showed that sentence-aware chunks produced cleaner evidence boundaries and stronger query-passage similarity on this initial test.

This limitation motivated the next project decision: evaluate Atlas against a larger corpus with more difficult retrieval cases.

---

## Dense Retrieval

Atlas currently uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to independently encode document chunks and user queries into dense vectors.

For a collection of document chunks:

```text
Chunk 1 ──→ embedding
Chunk 2 ──→ embedding
Chunk 3 ──→ embedding
...
```

These embeddings can be computed once and reused.

When a query arrives:

```text
Query ──→ embedding
             ↓
Compare with stored chunk embeddings
             ↓
Rank by similarity
```

Embeddings are normalized, allowing vector dot products to be used as cosine-similarity scores.

### Why dense retrieval?

Dense retrieval can identify passages that are semantically related even when the query and passage do not contain exactly the same words.

It also separates document indexing from query-time retrieval, making it possible to precompute document embeddings.

### Alternative: keyword retrieval

A traditional lexical retrieval system such as BM25 could instead rank documents based largely on matching terms.

This can be particularly effective when exact terminology matters and does not require neural embeddings.

Atlas currently uses dense retrieval because semantic matching is important for natural-language research questions. However, BM25 or a hybrid dense + lexical retrieval system is a potential future experiment rather than something being ruled out.

---

## Why Add Reranking?

Dense retrieval is efficient, but it has an important limitation.

The query and passage are encoded **separately**:

```text
Query ─────→ vector
                   \
                    similarity
                   /
Passage ───→ vector
```

This is useful for quickly finding potentially relevant passages, but the embedding model never jointly examines the question and passage.

Atlas therefore adds a second retrieval stage using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The cross-encoder receives:

```text
(query, candidate passage)
```

together and produces a new relevance score.

The resulting architecture is:

```text
Full document collection
          ↓
    Dense retrieval
          ↓
 Broad candidate set
          ↓
 Cross-encoder reranking
          ↓
 Highest-ranked evidence
```

### Why not use the cross-encoder for everything?

Cross-encoders can make more detailed query-passage comparisons, but they are substantially more expensive at query time.

With 100,000 chunks, dense retrieval can compare a query vector against precomputed document vectors.

Using only a cross-encoder would instead require evaluating approximately:

```text
(query, chunk 1)
(query, chunk 2)
...
(query, chunk 100,000)
```

for every new query.

Atlas therefore uses the dense retriever for candidate generation and the cross-encoder for more expensive analysis of only the strongest candidates.

---

## Initial Reranking Experiment

The current benchmark compares the original dense ranking against the cross-encoder ranking.

For example:

```text
Question:
"What do machine learning models learn?"

Dense retrieval:

0.729   Machine learning models learn patterns from data.
0.390   Language models generate text based on context.
0.302   RAG combines retrieval with generation.

Cross-encoder reranking:

 8.969  Machine learning models learn patterns from data.
-7.355  Language models generate text based on context.
-10.038 RAG combines retrieval with generation.
```

The two types of scores should not be compared numerically.

Dense scores represent similarity between normalized embedding vectors. Cross-encoder outputs are separate relevance scores used to rank candidate passages and are not probabilities.

The initial evaluation currently produces:

| Metric | Dense Retrieval | After Reranking |
|---|---:|---:|
| Hit@1 | 1.00 | 1.00 |
| Hit@3 | 1.00 | 1.00 |

Reranking therefore has **not yet demonstrated an improvement in Hit@k**.

The initial benchmark is already saturated, making improvement mathematically impossible on these metrics. The next evaluation stage will use more documents, distractor passages, and more difficult queries to determine when reranking actually helps.

---

## Evaluation

Atlas keeps evaluation separate from retrieval so that changes to the system can be measured against the same expected results.

Evaluation cases are stored in JSON and currently contain:

```json
{
    "query": "What does RAG combine?",
    "expected": "RAG combines retrieval with generation"
}
```

The current metrics are:

**Hit@1**

Was the expected evidence the first retrieved result?

**Hit@3**

Was the expected evidence anywhere within the first three results?

This creates a simple initial retrieval benchmark.

### Current evaluation limitation

Expected evidence is currently identified using text matching.

This is intentionally simple but brittle. A relevant passage may contain the correct information without containing the exact expected text.

Potential future evaluation approaches include:

- document or chunk IDs as ground truth
- Recall@k
- Mean Reciprocal Rank (MRR)
- manually labeled relevance judgments
- semantic answer evaluation
- LLM-assisted evaluation with human validation

The evaluation methodology will become more sophisticated as the corpus becomes more realistic.

---

## Project Structure

```text
Atlas/
├── data/
│   └── raw/
│       └── eval_cases.json
│
├── src/
│   └── atlas/
│       ├── __init__.py
│       ├── chunking.py
│       ├── retrieval.py
│       ├── reranking.py
│       ├── evaluation.py
│       └── demo.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The components are intentionally separated by responsibility:

```text
chunking.py
    ↓
How should documents be divided?

retrieval.py
    ↓
Which chunks might be relevant?

reranking.py
    ↓
Which retrieved candidates are most relevant?

evaluation.py
    ↓
How well did retrieval perform?

demo.py
    ↓
How do the components operate together?
```

This modular structure makes it possible to change one stage while keeping the others relatively constant during experiments.

---

## Running Atlas

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the current pipeline from the project root:

```bash
python -m src.atlas.demo
```

The current demo:

1. Divides source text into sentence-aware chunks.
2. Generates dense embeddings.
3. Retrieves candidate evidence using vector similarity.
4. Reranks candidates using a cross-encoder.
5. Compares dense and reranked results.
6. Calculates Hit@1 and Hit@3.

---

## Development Roadmap

Atlas is being developed incrementally, with evaluation performed before increasing system complexity.

### Current — Retrieval Foundation

- [x] Chunk representation and source metadata
- [x] Character-based chunking baseline
- [x] Sentence-aware chunking
- [x] Dense document embeddings
- [x] Vector similarity retrieval
- [x] Top-k ranking
- [x] External evaluation cases
- [x] Hit@1 and Hit@3 evaluation
- [x] Chunking experiment
- [x] Cross-encoder reranking
- [x] Dense vs. reranked comparison

### Next — Retrieval Evaluation at Larger Scale

- [ ] Load external source documents
- [ ] Expand the document corpus
- [ ] Add more difficult and ambiguous queries
- [ ] Introduce distractor passages
- [ ] Expand retrieval ground truth
- [ ] Add MRR and/or Recall@k
- [ ] Measure when reranking improves retrieval

Potential experiments include:

```text
Character vs. sentence vs. token chunking

Different chunk sizes

Dense vs. lexical retrieval

Dense vs. hybrid retrieval

Retrieval with vs. without reranking

Different candidate-set sizes
```

The goal is not to implement every option, but to use evaluation to determine which changes are justified.

### Planned — Grounded Generation

Once retrieval performance can be evaluated meaningfully:

```text
Question
    ↓
Retrieval
    ↓
Reranking
    ↓
Evidence
    ↓
LLM
    ↓
Grounded answer
```

Planned capabilities include:

- answer generation from retrieved context
- source attribution
- passage-level citations
- instructions to avoid unsupported claims
- handling cases where retrieved evidence is insufficient

### Planned — Answer Evaluation

Retrieval quality does not guarantee answer quality.

Atlas will therefore evaluate generated responses separately for properties such as:

```text
Relevance
    +
Grounding
    +
Citation support
    +
Factual consistency
```

Possible evaluation methods include deterministic checks, semantic similarity, LLM-based judges, and manually labeled evaluation cases.

### Future — Research Agent

The longer-term direction is to move from a single retrieval request toward multi-step research.

A future pipeline may resemble:

```text
User question
      ↓
Query classification
      ↓
Question decomposition
      ↓
Research planning
      ↓
Multiple retrieval operations
      ↓
Evidence reranking
      ↓
Evidence synthesis
      ↓
Grounded answer + citations
      ↓
Evaluation
```

This could allow Atlas to determine that a complex question requires multiple pieces of evidence rather than treating every request as a single vector search.

### Future — Adaptive Research and Tutoring

A further extension is to make Atlas responsive to the user's knowledge and research process.

Potential capabilities include:

- identifying missing evidence
- asking clarifying questions
- adapting explanations to the user's demonstrated understanding
- tracking unresolved research questions
- comparing conflicting sources
- identifying uncertainty
- revising retrieval strategies after weak results

These capabilities are intentionally downstream of retrieval and evaluation. The project first needs a reliable way to retrieve and measure evidence before adding autonomous behavior.

---

## Design Philosophy

Atlas follows several principles during development:

**Measure before assuming improvement.**  
New components should be evaluated against an existing baseline whenever possible.

**Prefer simple baselines first.**  
Character chunking and Hit@k are intentionally simple. More sophisticated approaches are introduced when their benefits can be tested.

**Keep components modular.**  
Chunking, retrieval, reranking, generation, and evaluation should be replaceable independently.

**Separate retrieval quality from generation quality.**  
A poor answer can result from failed retrieval, failed ranking, or failed generation. Evaluating these stages separately makes failures easier to diagnose.

**Document unsuccessful or neutral experiments.**  
A change does not need to improve a metric to provide useful information. For example, reranking currently preserves rather than improves Hit@k because the initial benchmark is saturated.

**Increase complexity only when justified.**  
More advanced techniques such as semantic chunking, hybrid search, agentic retrieval, or LLM-based evaluation are treated as hypotheses to test rather than automatic upgrades.

---

## Current Status

Atlas currently has a working end-to-end retrieval and reranking pipeline.

The immediate next milestone is moving from a four-sentence synthetic corpus to a larger document collection and more challenging evaluation benchmark. This will provide a meaningful baseline for determining whether reranking and subsequent retrieval improvements measurably improve the system.