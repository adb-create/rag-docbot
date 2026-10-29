"""Vector index: stores chunk vectors and returns the most similar ones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity in [0, 1]
    rank: int  # 1 = best match


def _as_1d(scores) -> np.ndarray:
    """Sparse (TF-IDF) and dense (neural) matrices both end up as a flat float array."""
    if hasattr(scores, "toarray"):
        scores = scores.toarray()
    return np.asarray(scores, dtype=float).ravel()


class VectorIndex:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix = None

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        if self.chunks:
            self._matrix = self.embedder.fit_transform([c.text for c in self.chunks])
        else:
            self._matrix = None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self._matrix is None or not query.strip():
            return []
        query_vector = self.embedder.transform([query])
        scores = _as_1d(self._matrix @ query_vector.T)
        order = np.argsort(-scores, kind="stable")[: min(k, len(scores))]
        return [Hit(self.chunks[i], float(scores[i]), rank) for rank, i in enumerate(order, 1)]
