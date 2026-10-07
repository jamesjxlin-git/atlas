import pytest

from src.atlas.chunking import split_sentences, split_text, split_tokens
from src.atlas.paper import identify_section_heading


def test_sentence_chunking_preserves_sentence():
    text = "Machine learning uses data. Retrieval finds evidence."
    chunks = split_sentences(text, source="test", size=30)
    assert chunks[0].text == "Machine learning uses data."
    assert chunks[1].text == "Retrieval finds evidence."
    for chunk in chunks:
        assert chunk.text == text[chunk.start:chunk.end]


def test_numbered_methods_heading():
    assert identify_section_heading("2. Methods") == "methods"


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1), (-1, 0)])
def test_fixed_chunking_rejects_nonprogressing_settings(size, overlap):
    with pytest.raises(ValueError):
        split_text("abc", "test", size, overlap)


def test_token_chunking_real_local_wordpiece_tokenizer():
    # Real fast tokenizer created locally: no model download or network.
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from transformers import PreTrainedTokenizerFast
    vocabulary = {"[UNK]": 0, "word": 1, ".": 2, "a": 3, "##b": 4}
    backend = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = BertPreTokenizer()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    text = " ".join(["word."] * 20) + " " + "ab" * 20
    chunks = split_tokens(text, "test", tokenizer, max_tokens=8, overlap_tokens=2)
    assert len(chunks) > 2
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start:chunk.end]
        assert len(tokenizer.encode(chunk.text, add_special_tokens=False)) <= 8
        covered.update(range(chunk.start, chunk.end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    assert chunks[-1].end == len(text)


def test_empty_text():
    assert split_sentences(" \n\t ", "empty") == []
    assert split_text("", "empty") == []
