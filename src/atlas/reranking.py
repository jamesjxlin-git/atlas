from sentence_transformers import CrossEncoder

from .chunking import Chunk


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        #Load a model trained to score query-document relevance
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[tuple[Chunk, float]],
        k: int = 3
    ) -> list[tuple[Chunk, float]]:

        #Pair the query with each retrieved chunk so they can be scored together
        pairs = [
            (query, piece.text)
            for piece, _ in results
        ]

        #Score how relevant each retrieved chunk is to the query
        scores = self.model.predict(pairs)

        #Attach each new score to its original chunk
        ranked = [
            (results[i][0], float(scores[i]))
            for i in range(len(results))
        ]

        #Sort by reranker score from highest to lowest
        ranked.sort(
            key=lambda result: result[1],
            reverse=True
        )

        return ranked[:k]