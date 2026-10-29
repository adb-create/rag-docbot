from docbot.chunker import Chunk
from docbot.embeddings import TfidfEmbedder, get_embedder
from docbot.retriever import VectorIndex

import pytest

TEXTS = [
    "Cats are small furry animals that purr.",
    "Python is a programming language used for data science.",
    "The Pacific Ocean is the largest ocean on Earth.",
]


def make_index() -> VectorIndex:
    index = VectorIndex(TfidfEmbedder())
    index.build([Chunk(text, "doc.txt", 1, i) for i, text in enumerate(TEXTS)])
    return index


def test_most_relevant_chunk_ranks_first():
    hits = make_index().search("which programming language is used for data science?")
    assert hits[0].chunk.chunk_id == 1


def test_hits_are_sorted_with_ranks_from_one():
    hits = make_index().search("ocean animals language")
    assert [h.rank for h in hits] == [1, 2, 3]
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_k_limits_the_number_of_results():
    index = make_index()
    assert len(index.search("ocean", k=2)) == 2
    assert len(index.search("ocean", k=10)) == 3


def test_empty_index_and_blank_query_return_nothing():
    assert VectorIndex(TfidfEmbedder()).search("anything") == []
    assert make_index().search("   ") == []


def test_unrelated_query_scores_zero():
    assert make_index().search("quantum chromodynamics")[0].score == 0.0


def test_unknown_embedder_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder("word2vec")
