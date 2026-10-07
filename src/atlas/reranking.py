"""Rescore retrieved query/passage pairs in a single batch."""

import numpy as np
from .chunking import Chunk


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
                 *, model=None, batch_size=32):
        if model is None:
            from .models import load_reranking_model
            model = load_reranking_model(model_name)
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size

    def rerank(self, query: str, results: list[tuple[Chunk, float]],
               k: int = 3) -> list[tuple[Chunk, float]]:
        if not query.strip() or k <= 0:
            raise ValueError("Provide a nonempty query and positive k.")
        if not results:
            return []
        scores = np.asarray(self.model.predict(
            [(query, piece.text) for piece, _ in results],
            batch_size=self.batch_size, show_progress_bar=False
        )).reshape(-1)
        if len(scores) != len(results) or not np.isfinite(scores).all():
            raise ValueError("Invalid reranker scores.")
        ranked = [(piece, float(score)) for (piece, _), score in zip(results, scores)]
        ranked.sort(key=lambda result: result[1], reverse=True)
        # These scores are ranking signals, not calibrated answer probabilities.
        return ranked[:k]
