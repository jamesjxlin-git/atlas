import re

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int


def split_text(
    text: str,
    source: str,
    size: int = 800,
    overlap: int = 120
) -> List[Chunk]:

    # Store completed chunks and track where the next chunk begins
    chunks = []
    start = 0

    # Create fixed-size chunks while preserving some text between neighboring chunks
    while start < len(text):
        end = min(start + size, len(text))

        piece = Chunk(
            text=text[start:end],
            source=source,
            start=start,
            end=end
        )

        chunks.append(piece)

        # Stop after reaching the end of the document
        if end == len(text):
            break

        # Move forward while retaining context from the previous chunk
        start = end - overlap

    return chunks


def split_sentences(
    text: str,
    source: str,
    size: int = 800
) -> List[Chunk]:

    # Find sentence-like units while preserving their positions in the original document
    matches = re.finditer(
        r".+?(?:[.!?](?=\s|$)|$)",
        text,
        re.DOTALL
    )

    sentences = []

    for match in matches:
        sentence = match.group().strip()

        # Ignore matches that contain only whitespace
        if not sentence:
            continue

        start = match.start()
        end = match.end()

        sentences.append((sentence, start, end))

    chunks = []
    current = []
    chunk_start = None
    chunk_end = None
    current_length = 0

    # Combine complete sentences until adding another would exceed the target size
    for sentence, start, end in sentences:
        sentence_length = len(sentence)

        if not current:
            chunk_start = start

        # Finish the current chunk before adding a sentence that would make it too large
        if current and current_length + 1 + sentence_length > size:
            chunks.append(
                Chunk(
                    text=" ".join(current),
                    source=source,
                    start=chunk_start,
                    end=chunk_end
                )
            )

            current = []
            current_length = 0
            chunk_start = start

        # Add the sentence to the chunk currently being built
        current.append(sentence)

        if current_length == 0:
            current_length = sentence_length
        else:
            current_length += 1 + sentence_length

        chunk_end = end

    # Save the final chunk because the loop ends before another sentence can trigger it
    if current:
        chunks.append(
            Chunk(
                text=" ".join(current),
                source=source,
                start=chunk_start,
                end=chunk_end
            )
        )

    return chunks