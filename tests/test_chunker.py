import re

import pytest

from docbot.chunker import chunk_pages, chunk_text

SENTENCES = [f"Sentence number {i} talks about topic {i}." for i in range(60)]
TEXT = " ".join(SENTENCES)


def test_chunks_never_exceed_size():
    chunks = chunk_text(TEXT, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_no_sentence_is_lost():
    joined = " ".join(chunk_text(TEXT, size=200, overlap=40))
    assert all(sentence in joined for sentence in SENTENCES)


def test_chunks_are_made_of_whole_sentences():
    for chunk in chunk_text(TEXT, size=200, overlap=40):
        assert chunk.startswith("Sentence number")
        assert chunk.endswith(".")


def test_consecutive_chunks_overlap():
    chunks = chunk_text(TEXT, size=200, overlap=60)
    for earlier, later in zip(chunks, chunks[1:]):
        last_sentence = earlier.rsplit(". ", 1)[-1]
        assert last_sentence in later


def test_zero_overlap_repeats_nothing():
    chunks = chunk_text(TEXT, size=200, overlap=0)
    found = re.findall(r"Sentence number \d+ talks about topic \d+\.", " ".join(chunks))
    assert len(found) == len(SENTENCES)


def test_long_sentence_without_punctuation_is_split_on_words():
    chunks = chunk_text("word " * 500, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_single_giant_token_is_sliced():
    chunks = chunk_text("x" * 1000, size=100, overlap=0)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert sum(len(chunk) for chunk in chunks) == 1000


def test_blank_text_gives_no_chunks():
    assert chunk_text("  \n\n  ") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_arguments_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world.", size=size, overlap=overlap)


def test_chunk_pages_keeps_source_page_and_ids():
    pages = [(1, "Alpha beta gamma."), (2, "Delta epsilon zeta.")]
    chunks = chunk_pages(pages, "notes.pdf", size=100, overlap=0)
    assert [(c.source, c.page, c.chunk_id) for c in chunks] == [
        ("notes.pdf", 1, 0),
        ("notes.pdf", 2, 1),
    ]
