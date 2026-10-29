"""Measure retrieval quality, so chunking/embedding choices are decided by numbers.

Each answerable question carries an ``evidence`` string: a phrase that must appear in
a retrieved chunk for that chunk to count as correct. Metrics:

* hit@k   : fraction of questions whose evidence appears in the top-k chunks
* MRR     : mean reciprocal rank of the first correct chunk (1.0 = always rank 1)
* abstain : fraction of *unanswerable* questions correctly refused (no relevant chunk)
* false abstain : fraction of answerable questions wrongly refused

Run:  python -m docbot.evaluate --docs data/sample_docs --questions data/eval_questions.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .pipeline import RAGPipeline

KS = (1, 3, 5)
BUDGET_CHARS = 1600  # about 400 tokens of context handed to the LLM


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval(
    docs_dir: str | Path,
    questions: list[dict],
    *,
    chunk_size: int,
    overlap: int,
    embedder: str = "tfidf",
    min_score: float | None = None,
    ks: tuple[int, ...] = KS,
    budget_chars: int = BUDGET_CHARS,
) -> dict:
    pipeline = RAGPipeline(
        embedder=embedder, llm="extractive", chunk_size=chunk_size, overlap=overlap,
        min_score=min_score,
    )
    pipeline.add_directory(docs_dir)
    pipeline.build()

    answerable = [q for q in questions if q.get("answerable", True)]
    unanswerable = [q for q in questions if not q.get("answerable", True)]

    ranks: list[int | None] = []
    false_abstain = 0
    budget_hits = 0
    for q in answerable:
        hits = pipeline.retrieve(q["question"], k=50)
        evidence = _norm(q["evidence"])
        rank = next((h.rank for h in hits if evidence in _norm(h.chunk.text)), None)
        ranks.append(rank)
        false_abstain += pipeline.should_abstain(hits)

        # Fair comparison across chunk sizes: give each config the same amount of
        # context (the best chunks that fit in `budget_chars`, at least one chunk).
        used, kept = 0, []
        for h in hits:
            if kept and used + len(h.chunk.text) > budget_chars:
                break
            kept.append(h)
            used += len(h.chunk.text)
        budget_hits += any(evidence in _norm(h.chunk.text) for h in kept)

    correct_abstain = sum(
        pipeline.should_abstain(pipeline.retrieve(q["question"], k=max(ks))) for q in unanswerable
    )

    n = len(answerable)
    result = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedder": embedder,
        "n_chunks": pipeline.n_chunks,
        "hit@budget": budget_hits / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "abstain_correct": correct_abstain / len(unanswerable) if unanswerable else None,
        "false_abstain": false_abstain / n,
    }
    for k in ks:
        result[f"hit@{k}"] = sum(1 for r in ranks if r and r <= k) / n
    result["misses"] = [q["question"] for q, r in zip(answerable, ranks) if r is None]
    return result


def sweep(docs_dir, questions, chunk_sizes=(200, 400, 800, 1200), overlap_ratio=0.15,
          embedder="tfidf", min_score=None) -> list[dict]:
    return [
        run_eval(docs_dir, questions, chunk_size=size, overlap=int(size * overlap_ratio),
                 embedder=embedder, min_score=min_score)
        for size in chunk_sizes
    ]


def to_markdown(results: list[dict], n_answerable: int, n_unanswerable: int) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    lines = [
        f"Retrieval quality on {n_answerable} answerable + {n_unanswerable} unanswerable questions.",
        f"`hit@{BUDGET_CHARS}ch` = answer found within the best chunks that fit in "
        f"{BUDGET_CHARS} characters of context (equal context for every chunk size).",
        "",
        f"| Embedder | Chunk size | Overlap | Chunks | hit@1 | hit@3 | hit@5 | hit@{BUDGET_CHARS}ch | MRR | Refuses unanswerable | Wrongly refuses |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['embedder']} | {r['chunk_size']} | {r['overlap']} | {r['n_chunks']} | "
            f"{pct(r['hit@1'])} | {pct(r['hit@3'])} | {pct(r['hit@5'])} | {pct(r['hit@budget'])} | "
            f"{r['mrr']:.3f} | {pct(r['abstain_correct'])} | {pct(r['false_abstain'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocBot retrieval quality")
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--embedder", default="tfidf")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[200, 400, 600, 800, 1200])
    parser.add_argument("--out", help="optional path to write the markdown table")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    results = sweep(args.docs, questions, tuple(args.chunk_sizes), embedder=args.embedder)
    n_un = sum(1 for q in questions if not q.get("answerable", True))
    table = to_markdown(results, len(questions) - n_un, n_un)
    print(table)
    for r in results:
        if r["misses"]:
            print(f"\nMissed at chunk_size={r['chunk_size']}:")
            for question in r["misses"]:
                print(f"  - {question}")
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
