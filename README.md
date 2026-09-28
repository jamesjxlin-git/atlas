# Atlas

Atlas is an evaluation-driven AI research assistant designed to retrieve, rank, summarize, and eventually synthesize evidence from source documents, with the broader goal of understanding how individual system-design and model choices affect retrieval, summarization, and eventually answer quality rather than beginning with a high-level RAG framework and treating each component as a black box.

Atlas currently implements sentence-aware document chunking, dense vector retrieval, cross-encoder reranking, retrieval evaluation, research-paper section parsing, and transformer-based summarization, while the summarization layer includes both an embedding-based extractive baseline and a pretrained abstractive transformer baseline implemented with PyTorch and Hugging Face Transformers.

Future stages will expand the evaluation framework, test the system on real research papers and larger corpora, measure factual consistency, introduce grounded generation and citations, and eventually explore model adaptation and multi-step research capabilities.

---

## Why Atlas?

A research assistant needs to do more than send a prompt to a language model because, before generating an answer, it needs to determine what information is relevant, retrieve appropriate evidence, distinguish strong evidence from loosely related passages, understand how a source is structured, and summarize the information in a way that remains faithful to the original material.

Atlas is therefore being developed as a modular system with two current foundations:

```text
                         Atlas
                           |
              ---------------------------
              |                         |
       Evidence Retrieval        Paper Understanding
              |                         |
       Parsing & Chunking          Section Parsing
              |                         |
           Embedding                 Summarization
              |                         |
       Dense Retrieval            Paper Overview
              |
     Candidate Passages
              |
   Cross-Encoder Reranking
              |
       Ranked Evidence
              |
      Grounded Generation        [planned]
              |
          Citations              [planned]
              |
      Answer Evaluation          [planned]
              |
   Feedback / Adaptation         [future]
```

Each stage is intentionally kept relatively independent so that alternative approaches can be tested without rebuilding the entire system, which also makes it easier to identify whether an improvement or failure originated from chunking, retrieval, reranking, summarization, or generation.

---

# Current Architecture

## 1. Document Chunking

Documents must be divided into smaller units before they can be embedded and retrieved, and Atlas initially used fixed-size character chunks with overlap because that approach is simple, deterministic, inexpensive, and easy to evaluate.

```text
Document
   ↓
Characters 0–100
Characters 80–180
Characters 160–260
...
```

However, character boundaries have no understanding of language, which meant words and sentences could be split between chunks and produce evidence such as:

```text
"nformation relevant to a query..."
```

or:

```text
"ing external sources."
```

### Current approach: sentence-aware chunking

Atlas now identifies sentence boundaries and groups complete sentences together until adding another sentence would exceed the target chunk size.

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

This approach produces more coherent evidence while still remaining lightweight, and the implementation also preserves source information along with starting and ending character positions so that the same metadata can later support source attribution and citations.

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

A more advanced approach could use embeddings or topic changes to identify when the meaning of a document shifts, but because this introduces additional computation and complexity, Atlas will only adopt it if later evaluation shows that the improvement justifies the added cost.

---

## Chunking Experiment

The first Atlas experiment compared fixed-size character chunking against sentence-aware chunking while keeping the embedding model, queries, and retrieval procedure constant so that chunking strategy remained the primary changed variable.

Sentence-aware chunking increased query-passage similarity for all six initial evaluation queries, with examples of top-result similarity changes including:

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

This does **not** establish that sentence-aware chunking improves retrieval accuracy because the initial six-query benchmark is small and the character baseline already retrieves the expected evidence for every query; however, the experiment did show that sentence-aware chunks produced cleaner evidence boundaries and stronger query-passage similarity on this initial test.

Because the initial benchmark is already saturated, the next retrieval evaluation stage will require a larger corpus with more difficult queries and distractor passages before any stronger claim can be made.

---

## 2. Dense Retrieval

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

These embeddings can be computed once and reused, while each incoming query is embedded separately and compared against the stored chunk representations.

```text
Query ──→ embedding
             ↓
Compare with stored chunk embeddings
             ↓
Rank by similarity
```

The embeddings are normalized, which allows vector dot products to be used as cosine-similarity scores.

### Why dense retrieval?

Dense retrieval is useful because it can identify passages that are semantically related even when the query and passage do not contain exactly the same words, while separating document indexing from query-time retrieval also makes it possible to precompute document embeddings instead of repeatedly encoding the same source material.

### Alternative: keyword retrieval

A traditional lexical retrieval system such as BM25 could instead rank documents based largely on matching terms, which can be particularly effective when exact terminology matters and does not require neural embeddings.

Atlas currently uses dense retrieval because semantic matching is important for natural-language research questions; however, BM25 or a hybrid dense + lexical retrieval system remains a potential future experiment rather than something being ruled out.

---

## 3. Cross-Encoder Reranking

Dense retrieval is efficient, but because the query and passage are encoded separately, the embedding model never directly examines their relationship as a pair.

```text
Query ─────→ vector
                 \
                  similarity
                 /
Passage ───→ vector
```

Atlas therefore adds a second retrieval stage using:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The cross-encoder receives:

```text
(query, candidate passage)
```

together and produces a new relevance score, which means it can evaluate the relationship between the query and candidate passage more directly than the dense retriever.

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

Cross-encoders can make more detailed query-passage comparisons, but they are substantially more expensive at query time because they must evaluate every query-passage pair individually.

With 100,000 chunks, dense retrieval can compare a query vector against precomputed document vectors, whereas using only a cross-encoder would require evaluating approximately:

```text
(query, chunk 1)
(query, chunk 2)
...
(query, chunk 100,000)
```

for every new query.

Atlas therefore uses the dense retriever for candidate generation and reserves the more expensive cross-encoder for evaluating only the strongest candidates.

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

The two types of scores should not be compared numerically because dense scores represent similarity between normalized embedding vectors, whereas cross-encoder outputs are separate relevance scores used to rank candidate passages and are not probabilities.

The initial evaluation currently produces:

| Metric | Dense Retrieval | After Reranking |
|---|---:|---:|
| Hit@1 | 1.00 | 1.00 |
| Hit@3 | 1.00 | 1.00 |

Reranking therefore has **not yet demonstrated an improvement in Hit@k**, but this result is largely a consequence of the initial benchmark already being saturated, which makes further improvement mathematically impossible on these metrics.

The next evaluation stage will therefore use more documents, distractor passages, and more difficult queries so that Atlas can measure when reranking actually improves retrieval rather than assuming that the additional model necessarily produces a better result.

---

## 4. Retrieval Evaluation

Atlas keeps evaluation separate from retrieval so that changes to the retrieval pipeline can be measured against the same expected results.

Evaluation cases are stored in JSON and currently contain:

```json
{
    "query": "What does RAG combine?",
    "expected": "RAG combines retrieval with generation"
}
```

The current metrics are:

### Hit@1

Was the expected evidence the first retrieved result?

### Hit@3

Was the expected evidence anywhere within the first three results?

This creates a simple initial retrieval benchmark, although expected evidence is currently identified using text matching, which is intentionally straightforward but can become brittle because a relevant passage may contain the correct information without containing the exact expected text.

Potential future evaluation approaches therefore include:

- document or chunk IDs as ground truth
- Recall@k
- Mean Reciprocal Rank (MRR)
- manually labeled relevance judgments
- semantic answer evaluation
- LLM-assisted evaluation with human validation

The evaluation methodology will become more sophisticated as the corpus becomes more realistic.

---

# Research Paper Understanding

Atlas now includes an initial research-paper understanding pipeline in addition to retrieval because a user may not know what specific questions to ask before they have at least a basic understanding of the paper.

The current pipeline begins with already-extracted research-paper text:

```text
Research Paper Text
        ↓
Section Detection
        ↓
Structured Paper Sections
        ↓
Section-Level Summarization
        ↓
Structured Paper Overview
```

PDF extraction is not yet part of the current pipeline, but once it is added, the same structured representation can be passed into the existing summarization system.

---

## 5. Research-Paper Section Parsing

Research papers are not treated as one undifferentiated block of text because different sections serve different purposes, and separating them allows later components to reason about Methods, Results, Discussion, and other sections independently.

Atlas currently identifies major sections such as:

```text
Abstract
Introduction
Methods
Results
Discussion
Limitations
Conclusion
```

Different papers may use different headings for similar concepts, so Atlas standardizes common variations.

For example:

```text
Methods
Methodology
Materials and Methods
Experimental Methods
```

are all treated internally as:

```text
methods
```

Similarly:

```text
Conclusion
Conclusions
```

are treated as the same section.

Section numbering is removed before matching, which means headings such as:

```text
2. Methods
3.1 Results
```

can still be recognized as:

```text
methods
results
```

The parser reads the paper line by line, identifies recognized headings, and assigns the following text to the appropriate standardized section, while parsing stops when the References or Bibliography section is reached because reference entries should not be treated as research-paper content for summarization.

The resulting representation resembles:

```python
{
    "abstract": "...",
    "introduction": "...",
    "methods": "...",
    "results": "...",
    "discussion": "...",
    "conclusion": "..."
}
```

This structure will later allow Atlas to treat sections differently depending on the user's question; for example, Methods can support methodology explanations while Results can contribute more heavily to identifying a paper's primary findings.

---

# 6. Summarization

Atlas currently implements two summarization approaches because keeping a simple baseline alongside a more complex model makes it possible to evaluate whether additional model complexity actually improves the result.

The first approach is an embedding-based extractive baseline, while the second is a pretrained transformer-based abstractive baseline.

---

## Extractive Summarization Baseline

The extractive approach does not generate new language; instead, it identifies existing sentences that are most representative of the overall section.

The pipeline is:

```text
Section
   ↓
Split into sentences
   ↓
Generate sentence embeddings
   ↓
Average sentence embeddings
   ↓
Approximate section-level semantic center
   ↓
Cosine similarity
   ↓
Select highest-scoring sentences
```

Suppose a section contains six sentences and the embedding model produces a tensor shaped approximately:

```text
[6, 384]
```

where:

```text
6   = number of sentences
384 = embedding dimensions
```

Averaging across the sentence dimension produces:

```text
[384]
```

which acts as an approximate semantic center for the section, while cosine similarity is then used to compare each sentence against that center and identify the most representative ones.

The selected sentences are returned in their original reading order because ranking them by similarity should not unnecessarily change the logical order of the source material.

### Why keep an extractive baseline?

Extractive summarization has an important advantage because the summary consists entirely of source text, which greatly reduces the possibility of inventing unsupported claims; however, it does not truly synthesize information.

For example, an extractive result may resemble:

```text
Students reported their average nightly sleep duration.

Students sleeping seven to nine hours had higher average exam scores.

Students sleeping fewer than five hours had the lowest average scores.
```

A more useful human-facing summary would ideally combine these ideas into a shorter explanation rather than simply selecting the original sentences, which is why the extractive method is currently treated as a baseline rather than the final user-facing approach.

---

## Transformer-Based Abstractive Summarization

Atlas also uses:

```text
sshleifer/distilbart-cnn-12-6
```

as its initial pretrained abstractive summarization baseline.

Unlike the extractive method, the transformer can generate new text by converting the source into token IDs, processing those numerical representations through a pretrained sequence-to-sequence transformer, and then decoding the generated token sequence back into readable text.

```text
Source Text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Pretrained Transformer
    ↓
Generated Token Sequence
    ↓
Decoded Summary
```

### Short sections

Very short sections are not forced through generative summarization because, when the source is already concise, asking the model to rewrite it creates additional opportunity for distortion without providing meaningful compression.

Atlas therefore preserves sufficiently short sections rather than generating a replacement unnecessarily.

### Long sections

Research-paper sections can exceed a model's practical input size, so Atlas uses token-aware hierarchical summarization rather than truncating everything beyond the model limit.

```text
Long Section
     ↓
Split into sentence-aware, token-limited chunks
     ↓
Summarize each chunk
     ↓
Combine chunk summaries
     ↓
Summarize combined information
     ↓
Final section summary
```

Sentence boundaries are preserved whenever possible; however, if a single sentence exceeds the model's safe input size, Atlas can fall back to token-level splitting so that the text can still be processed.

If the combined first-stage summaries remain too large, Atlas can apply another level of summarization before producing the final result, which allows the same pipeline to scale to substantially longer sections.

---

## PyTorch in the Current Pipeline

PyTorch is currently used directly for tensor operations and model inference, although Atlas has not yet reached the stage where model weights are trained or fine-tuned.

The extractive summarization pipeline uses operations such as:

```python
embeddings.mean(dim=0)
```

to calculate the average section embedding,

```python
torch.nn.functional.cosine_similarity(...)
```

to compare sentence embeddings against that section representation, and:

```python
torch.topk(...)
```

to identify the highest-ranked sentences.

The transformer summarization pipeline also runs a PyTorch BART model, while:

```python
with torch.no_grad():
```

disables gradient calculation because the current stage uses pretrained weights for inference rather than modifying them through training.

The current project therefore includes:

```text
PyTorch tensor operations          ✓
Embedding manipulation             ✓
Similarity computation             ✓
Pretrained transformer inference   ✓

Backpropagation                    not yet
Optimizer-based weight updates     not yet
Model training                     not yet
Fine-tuning                        not yet
```

Training and model adaptation remain later stages of Atlas, but the current implementation establishes the tensor, embedding, and transformer-inference foundation needed before those concepts are introduced.

---

## Initial Summarization Finding

Early testing demonstrated why generative summarization requires its own evaluation rather than being treated as automatically superior to extraction.

In one controlled example, the source stated:

```text
"We recruited 500 university students."
```

while the pretrained transformer initially introduced unsupported geographic information about those students.

The generated summary was more natural than the extractive baseline; however, because it included information that was not present in the source, the experiment showed that readability and factual consistency need to be evaluated as separate properties.

This motivates a central design requirement for later Atlas development:

```text
Readable generation
        +
Factual consistency
        +
Source grounding
```

The current DistilBART model is therefore treated as a **baseline rather than the final Atlas summarization model**, while later experiments will evaluate whether grounding, model adaptation, or fine-tuning can improve summary quality without increasing unsupported claims.

---

# Current Integrated Demo

The current `demo.py` exercises both major parts of Atlas so that retrieval and paper understanding can be tested in the same executable workflow without combining their evaluation prematurely.

### Retrieval pipeline

```text
Sample Document
      ↓
Sentence-Aware Chunking
      ↓
Dense Embeddings
      ↓
Dense Retrieval
      ↓
Candidate Passages
      ↓
Cross-Encoder Reranking
      ↓
Hit@1 / Hit@3 Evaluation
```

### Paper-understanding pipeline

```text
Sample Research Paper
      ↓
Section Parsing
      ↓
Abstract
Introduction
Methods
Results
Discussion
Conclusion
      ↓
Section-Level Summarization
      ↓
Structured Paper Overview
```

This allows Atlas to test retrieval and summarization separately while still confirming that the major components operate together correctly.

---

# Project Structure

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
│       ├── paper.py
│       ├── summarization.py
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

paper.py
    ↓
How should research papers be divided into meaningful sections?

summarization.py
    ↓
How should research-paper sections be condensed and compared across summarization approaches?

demo.py
    ↓
How do the components operate together?
```

This modular structure makes it possible to change one stage while keeping the others relatively constant during experiments, which is particularly important once Atlas begins comparing different models and retrieval strategies.

---

# Running Atlas

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
7. Parses a sample research paper into standardized sections.
8. Generates section-level summaries.
9. Produces a structured research-paper overview.

---

# Development Roadmap

Atlas is being developed incrementally, with evaluation performed before increasing system complexity so that new components are introduced because they solve an observed problem rather than simply because they are more advanced.

## Completed — Retrieval Foundation

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

---

## Completed — Paper Understanding Foundation

- [x] Research-paper section detection
- [x] Section-name normalization
- [x] Numbered-heading parsing
- [x] Extractive summarization baseline
- [x] Sentence embedding comparison
- [x] PyTorch-based tensor ranking
- [x] Transformer-based abstractive summarization
- [x] Token-aware long-section chunking
- [x] Hierarchical summarization
- [x] Pretrained transformer inference
- [x] Structured paper overview

---

## Next — Real Research Papers

The current paper-understanding pipeline has only been tested on controlled sample text, so the next milestone is to move from synthetic examples to real academic papers where formatting, section structure, terminology, and document length are substantially less predictable.

Planned work includes:

- [ ] Load real research-paper text
- [ ] Add PDF text extraction
- [ ] Test section detection across different paper formats
- [ ] Handle missing or unusual section headings
- [ ] Evaluate long Methods and Results sections
- [ ] Identify formatting artifacts from PDF extraction
- [ ] Evaluate summarization factual consistency
- [ ] Compare extractive and abstractive summaries
- [ ] Develop a structured paper-level synopsis

A target user-facing overview may eventually resemble:

```text
Research Question

Why It Matters

Methods

Main Findings

Limitations

Conclusion
```

This structure would allow users to understand a paper at a high level before deciding what more specific questions they want to ask.

---

## Next — Retrieval Evaluation at Larger Scale

The initial retrieval benchmark is intentionally small and currently saturated, so the next retrieval evaluation stage will make the task more difficult before Atlas attempts to claim that reranking or another retrieval modification improves performance.

Planned work includes:

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

The goal is not to implement every option, but rather to use evaluation to determine which changes are justified.

---

## Planned — Model Adaptation and Fine-Tuning

The current summarization model uses pretrained weights without modification, but a later Atlas stage will introduce actual model adaptation so that the project can move beyond inference and begin measuring whether changes to model parameters improve research-paper understanding.

The intended progression is:

```text
Pretrained Model
      ↓
Establish Baseline
      ↓
Atlas Training Data
      ↓
Fine-Tuning / Parameter-Efficient Adaptation
      ↓
Modified Model Weights
      ↓
Evaluation
      ↓
Compare Against Baseline
```

This stage will introduce concepts such as:

- training and validation data
- forward passes
- loss functions
- gradients
- backpropagation
- optimizers
- weight updates
- overfitting
- hyperparameter selection
- parameter-efficient fine-tuning
- model comparison

Potential methods include LoRA or other PEFT approaches, which would allow Atlas to adapt a pretrained transformer without requiring every parameter in the original model to be retrained.

The goal, however, is not to fine-tune a model simply to add another technique to the project; instead, model adaptation should be evaluated against the pretrained baseline to determine whether it measurably improves the tasks Atlas is designed to perform.

---

## Planned — Grounded Generation

Once retrieval and paper-understanding performance can be evaluated meaningfully, the two branches can begin operating together.

```text
Question
    ↓
Retrieval
    ↓
Reranking
    ↓
Evidence
    ↓
Generation Model
    ↓
Grounded Answer
```

Planned capabilities include:

- answer generation from retrieved context
- source attribution
- passage-level citations
- instructions to avoid unsupported claims
- handling cases where retrieved evidence is insufficient

A future paper workflow could therefore resemble:

```text
Research Paper
      ↓
Structured Overview
      ↓
User asks follow-up question
      ↓
Dense Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Relevant Evidence
      ↓
Grounded Answer
```

This structure allows Atlas to provide enough context for the user to understand the paper first, rather than requiring the user to know exactly what they want to ask before they have seen a synopsis.

---

## Planned — Answer and Summary Evaluation

Retrieval quality does not guarantee generation quality, while a fluent summary does not guarantee factual accuracy; therefore, Atlas will evaluate generated outputs separately from retrieval performance.

Potential evaluation dimensions include:

```text
Relevance
    +
Grounding
    +
Factual Consistency
    +
Citation Support
```

Possible methods include:

- deterministic checks
- semantic similarity
- source-to-summary consistency checks
- manually labeled evaluation cases
- LLM-based judges with human validation
- comparison against reference summaries

The current extractive and transformer summarizers provide the first two baselines for these later experiments.

---

## Future — Research Agent

The longer-term direction is to move from a single retrieval request toward multi-step research, where Atlas can determine that a complex question requires several pieces of evidence rather than treating every request as a single vector search.

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

---

## Future — Adaptive Research and Tutoring

A further extension is to make Atlas responsive to the user's knowledge and research process, but these capabilities are intentionally downstream of retrieval, document understanding, generation, and evaluation because the project first needs reliable ways to retrieve, summarize, generate, and measure evidence before introducing autonomous behavior.

Potential capabilities include:

- identifying missing evidence
- asking clarifying questions
- adapting explanations to the user's demonstrated understanding
- tracking unresolved research questions
- comparing conflicting sources
- identifying uncertainty
- revising retrieval strategies after weak results

---

# Design Philosophy

Atlas follows several principles during development.

### Measure before assuming improvement.

New components should be evaluated against an existing baseline whenever possible, which is why the project currently retains both extractive and abstractive summarization approaches rather than assuming that the generative model is automatically superior.

### Prefer simple baselines first.

Character chunking, Hit@k, and extractive summarization are intentionally simple because they create reference points that more sophisticated approaches can later be compared against.

### Keep components modular.

Chunking, retrieval, reranking, parsing, summarization, generation, and evaluation should remain replaceable independently so that changing one stage does not require rebuilding the entire system.

### Separate retrieval quality from generation quality.

A poor answer can result from failed retrieval, failed ranking, failed summarization, or failed generation, so evaluating these stages separately makes failures easier to diagnose and prevents one strong component from masking another weak one.

### Treat fluency and factual accuracy as different properties.

A generated response may sound better while being less faithful to the source, which the initial transformer summarization experiment demonstrated when the model produced more natural text but also introduced unsupported information.

### Document unsuccessful or neutral experiments.

A change does not need to improve a metric to provide useful information because, for example, reranking currently preserves rather than improves Hit@k because the initial benchmark is saturated, while the pretrained abstractive summarizer produces more natural language but can also introduce unsupported claims.

Both outcomes help determine what Atlas should test next.

### Increase complexity only when justified.

More advanced techniques such as semantic chunking, hybrid search, fine-tuning, agentic retrieval, or LLM-based evaluation are treated as hypotheses to test rather than automatic upgrades, because additional complexity is only useful if it produces a measurable improvement or solves a problem that the simpler baseline cannot.

---

# Current Status

Atlas currently has two integrated foundations: an evaluation-driven retrieval pipeline using sentence-aware chunking, dense retrieval, cross-encoder reranking, and Hit@k evaluation, along with an initial research-paper understanding pipeline using section parsing, extractive summarization, pretrained transformer-based abstractive summarization, PyTorch tensor operations, and hierarchical long-section summarization.

Both pipelines currently operate on controlled test inputs, so the immediate next milestone is to move Atlas from synthetic examples to real research papers and larger document collections, which will make it possible to evaluate retrieval quality, reranking quality, section-parsing reliability, summary quality, factual consistency, and eventually model adaptation before adding increasingly complex generation and agentic behavior.