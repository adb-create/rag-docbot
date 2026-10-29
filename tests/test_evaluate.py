import re

from docbot.evaluate import load_questions, run_eval, sweep, to_markdown

from conftest import QUESTIONS_FILE, SAMPLE_DIR


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def test_question_set_is_well_formed():
    questions = load_questions(QUESTIONS_FILE)
    texts = [q["question"] for q in questions]
    assert len(texts) == len(set(texts)), "duplicate questions"
    assert sum(1 for q in questions if q.get("answerable", True)) >= 25
    assert sum(1 for q in questions if not q.get("answerable", True)) >= 5


def test_every_evidence_phrase_exists_in_exactly_the_corpus():
    """A typo in `evidence` would silently count as a retrieval failure, so verify it."""
    corpus = _norm(" ".join(p.read_text(encoding="utf-8") for p in SAMPLE_DIR.glob("*.txt")))
    for q in load_questions(QUESTIONS_FILE):
        if q.get("answerable", True):
            assert _norm(q["evidence"]) in corpus, q["question"]


def test_retrieval_quality_does_not_regress():
    """Floors sit a little below the measured numbers (see docs/eval_results.md)."""
    result = run_eval(SAMPLE_DIR, load_questions(QUESTIONS_FILE), chunk_size=600, overlap=90)
    assert result["hit@budget"] >= 0.90
    assert result["hit@5"] >= 0.90
    assert result["mrr"] >= 0.80
    assert result["false_abstain"] == 0.0  # never refuse a question the notes can answer
    assert result["abstain_correct"] >= 0.80


def test_sweep_and_markdown_report():
    questions = load_questions(QUESTIONS_FILE)
    results = sweep(SAMPLE_DIR, questions, chunk_sizes=(300, 600))
    assert [r["chunk_size"] for r in results] == [300, 600]

    table = to_markdown(results, n_answerable=28, n_unanswerable=6)
    assert table.count("| tfidf |") == 2
    assert "hit@1600ch" in table
