"""The RAG pipeline: load -> chunk -> embed/index -> retrieve -> (guard) -> generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, chunk_pages
from .embeddings import get_embedder
from .llm import get_llm
from .loader import Pages, list_documents, load_path, pages_from_bytes
from .retriever import Hit, VectorIndex

NOT_FOUND_MESSAGE = "I couldn't find anything about that in the loaded documents."


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    abstained: bool = False  # True when retrieval found nothing relevant enough


class RAGPipeline:
    def __init__(
        self,
        embedder="tfidf",
        llm="extractive",
        chunk_size: int = 600,
        overlap: int = 90,
        top_k: int = 4,
        min_score: float | None = None,
    ) -> None:
        self.embedder = get_embedder(embedder) if isinstance(embedder, str) else embedder
        self.llm = get_llm(llm) if isinstance(llm, str) else llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        # Below this similarity the question is treated as "not in the documents"
        # and the LLM is never called. Cheaper, and stops confident hallucinations.
        self.min_score = self.embedder.default_min_score if min_score is None else min_score
        self._documents: dict[str, list[Chunk]] = {}
        self._index = VectorIndex(self.embedder)
        self._dirty = False

    # ------------------------------------------------------------------ ingestion

    def add_pages(self, source: str, pages: Pages) -> int:
        """Index a document. Re-adding the same file name replaces the old version."""
        chunks = chunk_pages(pages, source, self.chunk_size, self.overlap)
        self._documents[source] = chunks
        self._dirty = True
        return len(chunks)

    def add_bytes(self, name: str, data: bytes) -> int:
        return self.add_pages(name, pages_from_bytes(name, data))

    def add_file(self, path: str | Path) -> int:
        return self.add_pages(Path(path).name, load_path(path))

    def add_directory(self, folder: str | Path) -> int:
        return sum(self.add_file(path) for path in list_documents(folder))

    def remove(self, source: str) -> None:
        if self._documents.pop(source, None) is not None:
            self._dirty = True

    def build(self) -> None:
        if self._dirty:
            self._index.build([c for chunks in self._documents.values() for c in chunks])
            self._dirty = False

    # ------------------------------------------------------------------ querying

    @property
    def sources(self) -> list[str]:
        return sorted(self._documents)

    @property
    def n_chunks(self) -> int:
        return sum(len(chunks) for chunks in self._documents.values())

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        self.build()
        return self._index.search(question, k or self.top_k)

    def should_abstain(self, hits: list[Hit]) -> bool:
        return not hits or hits[0].score < self.min_score

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if self.should_abstain(hits):
            # Keep the closest passages so the UI can show what was (not) matched.
            return Answer(question, NOT_FOUND_MESSAGE, hits, abstained=True)
        # Only relevant passages reach the LLM: less noise, fewer tokens, and the
        # cited sources are exactly the passages the answer was built from.
        relevant = [hit for hit in hits if hit.score >= self.min_score]
        return Answer(question, self.llm.answer(question, relevant).strip(), relevant)
