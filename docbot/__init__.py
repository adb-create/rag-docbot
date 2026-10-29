"""DocBot: a small, well-tested Retrieval-Augmented Generation (RAG) pipeline."""

from .pipeline import NOT_FOUND_MESSAGE, Answer, RAGPipeline
from .retriever import Hit

__all__ = ["RAGPipeline", "Answer", "Hit", "NOT_FOUND_MESSAGE"]
__version__ = "0.1.0"
