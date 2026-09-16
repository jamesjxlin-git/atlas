def hit_at_k(results, expected_text: str, k: int = 3) -> int:
    top_results = results[:k]

    for piece, score in top_results:
        if expected_text.lower() in piece.text.lower():
            return 1
    return 0