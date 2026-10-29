"""Command-line interface for DocBot.

    python cli.py ask  data/sample_docs "What is thrashing?"
    python cli.py chat data/sample_docs --llm ollama
    python cli.py eval
"""

from __future__ import annotations

import argparse
import sys

from docbot import RAGPipeline
from docbot import evaluate


def _build(args) -> RAGPipeline:
    pipeline = RAGPipeline(
        embedder=args.embedder,
        llm=args.llm,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )
    n = pipeline.add_directory(args.docs)
    print(f"Indexed {len(pipeline.sources)} documents ({n} chunks).", file=sys.stderr)
    return pipeline


def _print_answer(answer) -> None:
    print(f"\n{answer.text}\n")
    label = "Closest passages (not used)" if answer.abstained else "Sources"
    for hit in answer.hits:
        print(f"  [{hit.rank}] {hit.chunk.source} p.{hit.chunk.page}  (score {hit.score:.2f})")
    if answer.hits:
        print(f"  ^ {label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docbot", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("docs", help="folder with .pdf/.txt/.md files")
        p.add_argument("--llm", default="extractive", choices=["extractive", "anthropic", "ollama"])
        p.add_argument("--embedder", default="tfidf", choices=["tfidf", "sentence-transformers"])
        p.add_argument("--chunk-size", type=int, default=600)
        p.add_argument("--overlap", type=int, default=90)
        p.add_argument("--top-k", type=int, default=4)

    ask = sub.add_parser("ask", help="answer one question")
    common(ask)
    ask.add_argument("question")

    chat = sub.add_parser("chat", help="interactive question loop")
    common(chat)

    sub.add_parser("eval", help="run the retrieval evaluation sweep")

    args = parser.parse_args(argv)

    if args.command == "eval":
        evaluate.main([])
        return 0

    try:
        pipeline = _build(args)
        if args.command == "ask":
            _print_answer(pipeline.ask(args.question))
        else:
            print("Ask a question (empty line to quit).")
            while True:
                question = input("\n> ").strip()
                if not question:
                    break
                _print_answer(pipeline.ask(question))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
