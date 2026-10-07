"""Connect ingestion, chunking, retrieval, reranking, context, and generation."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .chunking import Chunk, split_tokens
from .config import AtlasConfig
from .context import Passage, pack_context
from .generation import Generator, GenerationResult
from .ingestion import Document, load_documents
from .retrieval import Retriever
from .reranking import Reranker
from .grades import chunk_id


@dataclass
class SearchResult:
    question: str
    dense: list[tuple[Chunk, float]]
    reranked: list[tuple[Chunk, float]]
    passages: list[Passage]
    timings_ms: dict[str, float]
    generation: GenerationResult | None = None

    def to_dict(self):
        def results(items):
            return [{"chunk_id": chunk_id(chunk), "source": chunk.source, "start": chunk.start, "end": chunk.end,
                     "text": chunk.text, "score": score} for chunk, score in items]
        return {
            "question": self.question,
            "generation": self.generation.to_dict() if self.generation else None,
            "context": [
                {"id": p.id, "source": p.chunk.source, "start": p.chunk.start,
                 "end": p.chunk.end, "text": p.chunk.text, "rerank_score": p.score}
                for p in self.passages
            ],
            "dense": results(self.dense), "reranked": results(self.reranked),
            "timings_ms": self.timings_ms
        }


class Atlas:
    def __init__(self, config: AtlasConfig | None = None, *, cache_dir=None,
                 retriever=None, reranker=None, generator=None, encoding=None):
        self.config = config or AtlasConfig()
        self.retriever = retriever or Retriever(
            self.config.embedding_model, cache_dir=cache_dir
        )
        self.reranker = reranker or Reranker(self.config.reranking_model)
        self.generator = generator or Generator(
            self.config.generation_model, max_output_tokens=self.config.max_output_tokens
        )
        if encoding is None:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(self.config.generation_model)
            except KeyError:
                raise ValueError(
                    "No known tokenizer for this generation model. Supply an encoding "
                    "explicitly when constructing Atlas."
                ) from None
        self.encoding = encoding
        self.index_ms = 0.0

    def index_documents(self, documents: list[Document]):
        start = perf_counter()
        tokenizer = self.retriever.model.tokenizer
        safe_limit = (self.retriever.model.max_seq_length
                      - tokenizer.num_special_tokens_to_add(pair=False))
        budget = min(self.config.chunk_tokens, safe_limit)
        if self.config.overlap_tokens >= budget:
            raise ValueError("Chunk overlap exceeds this embedding model's safe input budget.")
        chunks = [
            chunk for document in documents
            for chunk in split_tokens(document.text, document.source, tokenizer,
                                      budget, self.config.overlap_tokens)
        ]
        self.retriever.index(chunks)
        self.index_ms = (perf_counter() - start) * 1000

    def index_path(self, path: str | Path):
        self.index_documents(load_documents(path))

    def search(self, question: str) -> SearchResult:
        tokenizer = self.retriever.model.tokenizer
        query_limit = min(64, self.retriever.model.max_seq_length
                          - tokenizer.num_special_tokens_to_add(pair=False))
        if not question.strip() or len(tokenizer.encode(
                question, add_special_tokens=False)) > query_limit:
            raise ValueError(f"Use a nonempty question of at most {query_limit} embedding tokens.")
        start = perf_counter()
        dense = self.retriever.search(question, k=self.config.candidate_k)
        retrieved_at = perf_counter()
        reranked = self.reranker.rerank(question, dense, k=len(dense))
        reranked_at = perf_counter()
        passages = pack_context(
            reranked, self.encoding, self.config.context_tokens, self.config.context_k
        )
        finished = perf_counter()
        return SearchResult(question, dense, reranked, passages, {
            "retrieval": (retrieved_at - start) * 1000,
            "reranking": (reranked_at - retrieved_at) * 1000,
            "context": (finished - reranked_at) * 1000,
        })

    def generate(self, result: SearchResult) -> SearchResult:
        start = perf_counter()
        result.generation = self.generator.generate(result.question, result.passages)
        result.timings_ms["generation"] = (perf_counter() - start) * 1000
        result.timings_ms["total"] = sum(
            value for key, value in result.timings_ms.items() if key != "total"
        )
        return result

    def ask(self, question: str) -> SearchResult:
        return self.generate(self.search(question))
