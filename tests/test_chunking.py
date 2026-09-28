def test_sentence_chunking_preserves_sentence():
    text = "Machine learning uses data. Retrieval finds evidence."

    chunks = split_sentences(
        text,
        source="test",
        size=100
    )

    assert chunks[0].text == "Machine learning uses data."

def test_numbered_methods_heading():
    assert identify_section_heading("2. Methods") == "methods"