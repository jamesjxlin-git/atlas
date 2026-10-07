from dataclasses import dataclass


@dataclass(frozen=True)
class AtlasConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    generation_model: str = "gpt-4.1-mini-2025-04-14"
    chunk_tokens: int = 180
    overlap_tokens: int = 32
    candidate_k: int = 20
    context_k: int = 4
    context_tokens: int = 3500
    max_output_tokens: int = 1600

    def __post_init__(self):
        if not 0 <= self.overlap_tokens < self.chunk_tokens:
            raise ValueError("Require overlap_tokens < chunk_tokens.")
        if not 0 < self.context_k <= self.candidate_k:
            raise ValueError("Require 0 < context_k <= candidate_k.")
        if self.context_tokens < 100 or self.max_output_tokens < 100:
            raise ValueError("Token budgets must be at least 100.")
