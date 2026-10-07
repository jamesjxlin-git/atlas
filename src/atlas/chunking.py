"""Chunk source text without losing its character offsets."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    start: int
    end: int


def split_text(text: str, source: str, size: int = 800,
               overlap: int = 120) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if text[start:end].strip():
            chunks.append(Chunk(text[start:end], source, start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def split_sentences(text: str, source: str, size: int = 800) -> list[Chunk]:
    """Character-budget baseline; long individual sentences may exceed size."""
    if size <= 0:
        raise ValueError("size must be positive.")
    spans = [(m.start(), m.end()) for m in
             re.finditer(r"\S.*?(?:[.!?](?=\s|$)|$)", text, re.DOTALL)]
    chunks = []
    start = end = None
    for sentence_start, sentence_end in spans:
        if start is not None and sentence_end - start > size:
            chunks.append(Chunk(text[start:end], source, start, end))
            start = None
        if start is None:
            start = sentence_start
        end = sentence_end
    if start is not None:
        chunks.append(Chunk(text[start:end], source, start, end))
    return chunks


def split_tokens(text: str, source: str, tokenizer,
                 max_tokens: int = 180, overlap_tokens: int = 32) -> list[Chunk]:
    """Use the embedding model's fast tokenizer; prefer nearby sentence ends.

    Tokens here exclude special tokens. Offsets refer to original text,
    including PDF page extraction text when the source is a PDF page.
    """
    if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Require max_tokens > 0 and overlap_tokens < max_tokens.")
    if not text.strip():
        return []
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Token chunking requires a fast tokenizer with offsets.")
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
        )["offset_mapping"]
    offsets = [(a, b) for a, b in offsets if b > a]
    sentence_ends = {m.end() for m in re.finditer(r"[.!?](?=\s|$)", text)}
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        if end < len(offsets):
            boundaries = [i + 1 for i in range(start + max_tokens // 2, end)
                          if offsets[i][1] in sentence_ends]
            if boundaries:
                end = boundaries[-1]
        # Retokenizing a substring can change boundary tokenization. Check it.
        while end > start:
            a, b = offsets[start][0], offsets[end - 1][1]
            if len(tokenizer.encode(text[a:b], add_special_tokens=False)) <= max_tokens:
                break
            end -= 1
        if end == start:
            raise ValueError("A token cannot fit the configured chunk budget.")
        chunks.append(Chunk(text[a:b], source, a, b))
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks
