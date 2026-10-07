"""Pack complete evidence passages under a generation-token budget."""

import json
from dataclasses import dataclass
from .chunking import Chunk


@dataclass(frozen=True)
class Passage:
    id: int
    chunk: Chunk
    score: float


def serialize_context(passages: list[Passage]) -> str:
    return json.dumps({"passages": [
        {"id": passage.id, "text": passage.chunk.text}
        for passage in passages
    ]}, ensure_ascii=False)


def pack_context(results: list[tuple[Chunk, float]], encoding,
                 max_tokens: int = 3500, max_passages: int = 4) -> list[Passage]:
    if max_tokens <= 0 or max_passages <= 0:
        raise ValueError("Context limits must be positive.")
    passages, seen = [], set()
    for chunk, score in results:
        fingerprint = " ".join(chunk.text.split())
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidate = passages + [Passage(len(passages) + 1, chunk, score)]
        if len(encoding.encode(serialize_context(candidate), disallowed_special=())) > max_tokens:
            continue
        passages = candidate
        if len(passages) == max_passages:
            break
    return passages
