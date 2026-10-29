"""Text -> vector backends.

Two interchangeable embedders share one tiny interface (`fit_transform`, `transform`):

* ``TfidfEmbedder``: lexical, needs no model download, works offline. Good baseline.
* ``SentenceTransformerEmbedder``: semantic, matches meaning even with different
  wording ("car" ~ "automobile"). Needs ``pip install sentence-transformers``.

All vectors are L2-normalised, so a dot product equals cosine similarity.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    name = "tfidf"
    # Cosine scores below this mean "nothing relevant". 0.05 sits just under the lowest
    # score of any genuine question in data/eval_questions.json (0.079): it never refuses
    # a real question, at the price of letting topically-adjacent questions through.
    default_min_score = 0.05

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )

    def fit_transform(self, texts: list[str]):
        try:
            return self._vectorizer.fit_transform(texts)
        except ValueError as exc:  # sklearn: "empty vocabulary"
            raise ValueError(
                "The documents contain no indexable text. "
                "Scanned PDFs need OCR before they can be searched."
            ) from exc

    def transform(self, texts: list[str]):
        return self._vectorizer.transform(texts)


class SentenceTransformerEmbedder:
    name = "sentence-transformers"
    # Starting point for cosine scores of MiniLM; re-calibrate with `python cli.py eval`.
    default_min_score = 0.30

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic embeddings need sentence-transformers: "
                "pip install -r requirements-optional.txt"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


EMBEDDERS = {
    "tfidf": TfidfEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}


def get_embedder(name: str):
    try:
        return EMBEDDERS[name]()
    except KeyError:
        raise ValueError(f"Unknown embedder '{name}'. Choose from: {', '.join(EMBEDDERS)}") from None
