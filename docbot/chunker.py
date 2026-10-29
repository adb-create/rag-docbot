"""Sentence-aware text chunking with overlap.

Why not just slice every N characters? Because that cuts sentences in half, which
hurts both retrieval (the embedding of half a sentence is noisy) and generation
(the LLM sees a truncated fact). Here chunks are built from whole sentences, and
the tail of each chunk is repeated at the start of the next one (the overlap) so a
fact that sits on a boundary still appears intact in at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document, with enough metadata to cite it."""

    text: str
    source: str  # file name
    page: int  # 1-based page number (always 1 for .txt/.md)
    chunk_id: int  # index of the chunk within its source file


def _hard_split(text: str, size: int) -> list[str]:
    """Split a single over-long sentence on word boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        # A single "word" longer than the limit (e.g. a base64 blob): slice it.
        while len(word) > size:
            if current:
                pieces.append(" ".join(current))
                current, length = [], 0
            pieces.append(word[:size])
            word = word[size:]
        extra = len(word) + (1 if current else 0)
        if current and length + extra > size:
            pieces.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += extra
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_units(text: str, size: int) -> list[str]:
    """Break text into sentence-sized units, none longer than `size`."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= size:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, size))
    return units


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split `text` into chunks of at most `size` characters.

    Consecutive chunks share up to `overlap` characters of trailing sentences.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in _split_units(text, size):
        extra = len(unit) + (1 if current else 0)
        if current and length + extra > size:
            chunks.append(" ".join(current))

            # Carry the last few sentences forward as overlap.
            carry: list[str] = []
            carried = 0
            for previous in reversed(current):
                cost = len(previous) + (1 if carry else 0)
                if carried + cost > overlap:
                    break
                carry.insert(0, previous)
                carried += cost
            current = carry
            length = carried

            # If the carried text plus the new unit is still too big, drop the carry.
            extra = len(unit) + (1 if current else 0)
            if current and length + extra > size:
                current, length = [], 0
                extra = len(unit)

        current.append(unit)
        length += extra

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]], source: str, size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Chunk a document page by page so every chunk keeps its page number."""
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for piece in chunk_text(text, size, overlap):
            chunks.append(Chunk(piece, source, page_number, len(chunks)))
    return chunks
