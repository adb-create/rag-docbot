import pytest

from docbot import NOT_FOUND_MESSAGE, RAGPipeline


class RecordingLLM:
    name = "recording"

    def __init__(self):
        self.calls = []

    def answer(self, question, hits):
        self.calls.append((question, hits))
        return "  stub answer  "


DOCS = {
    "cats.txt": "Cats purr when they are content. Cats sleep up to sixteen hours a day.",
    "python.txt": "Python is a programming language. It uses indentation to mark code blocks.",
}


def make_pipeline(llm=None, **kwargs) -> RAGPipeline:
    pipeline = RAGPipeline(llm=llm or RecordingLLM(), chunk_size=200, overlap=0, **kwargs)
    for name, text in DOCS.items():
        pipeline.add_bytes(name, text.encode())
    return pipeline


def test_answer_is_grounded_in_the_relevant_document():
    llm = RecordingLLM()
    answer = make_pipeline(llm).ask("How many hours do cats sleep?")

    assert answer.text == "stub answer"  # LLM output is stripped
    assert not answer.abstained
    assert answer.hits[0].chunk.source == "cats.txt"
    assert llm.calls and llm.calls[0][1] == answer.hits


def test_only_relevant_passages_reach_the_llm():
    llm = RecordingLLM()
    answer = make_pipeline(llm).ask("How many hours do cats sleep?")
    assert all(h.score >= 0.05 for h in answer.hits)
    assert all(h.chunk.source != "python.txt" for h in llm.calls[0][1])


def test_unrelated_question_abstains_without_calling_the_llm():
    llm = RecordingLLM()
    answer = make_pipeline(llm).ask("How tall is Mount Everest?")

    assert answer.abstained
    assert answer.text == NOT_FOUND_MESSAGE
    assert llm.calls == []


def test_strict_min_score_forces_abstention():
    answer = make_pipeline(min_score=0.99).ask("How many hours do cats sleep?")
    assert answer.abstained


def test_empty_pipeline_abstains():
    answer = RAGPipeline(llm=RecordingLLM()).ask("anything at all")
    assert answer.abstained and answer.hits == []


def test_re_adding_a_file_replaces_its_old_content():
    pipeline = make_pipeline()
    pipeline.add_bytes("cats.txt", b"Dogs bark loudly at strangers.")

    assert pipeline.sources == ["cats.txt", "python.txt"]
    assert pipeline.ask("Why do dogs bark?").hits[0].chunk.source == "cats.txt"
    assert pipeline.ask("How long do cats sleep?").abstained


def test_remove_drops_a_document():
    pipeline = make_pipeline()
    pipeline.remove("python.txt")
    assert pipeline.sources == ["cats.txt"]
    assert pipeline.ask("What is Python indentation?").abstained


def test_documents_added_after_a_search_are_searchable():
    pipeline = make_pipeline()
    pipeline.ask("cats")  # builds the index once
    pipeline.add_bytes("ocean.txt", b"The Pacific is the largest ocean on Earth.")
    assert pipeline.retrieve("largest ocean")[0].chunk.source == "ocean.txt"


def test_text_without_indexable_words_gives_a_clear_error():
    pipeline = RAGPipeline(llm=RecordingLLM())
    pipeline.add_bytes("empty.txt", b"the and of. it is.")
    with pytest.raises(ValueError, match="no indexable text"):
        pipeline.build()


def test_pdf_upload_is_searchable(make_pdf):
    pipeline = RAGPipeline(llm=RecordingLLM())
    assert pipeline.add_bytes("facts.pdf", make_pdf("Octopuses have three hearts")) == 1
    hit = pipeline.retrieve("how many hearts do octopuses have")[0]
    assert hit.chunk.source == "facts.pdf" and hit.chunk.page == 1


def test_end_to_end_on_the_sample_notes(sample_dir):
    pipeline = RAGPipeline()  # defaults: TF-IDF + extractive answers, fully offline
    pipeline.add_directory(sample_dir)

    assert len(pipeline.sources) == 4
    answer = pipeline.ask("What is thrashing?")
    assert "thrashing" in answer.text.lower()
    assert answer.hits[0].chunk.source == "operating_systems.txt"
