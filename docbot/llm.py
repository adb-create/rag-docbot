"""Answer generation backends.

Every backend implements ``answer(question, hits) -> str``.

* ``ExtractiveLLM``: no model at all; returns the best-matching sentences from the
  retrieved chunks. Lets the whole pipeline run (and be tested) offline.
* ``AnthropicLLM``: Claude through the Anthropic API (``ANTHROPIC_API_KEY``).
* ``OllamaLLM``: any local model served by Ollama, no API key or internet needed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from .retriever import Hit

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided by the user. "
    "Cite the passages you rely on with their numbers, like [1] or [2][3]. "
    "If the passages do not contain the answer, reply that you could not find it in the "
    "documents. Never use outside knowledge and never invent citations."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Format retrieved chunks as numbered, source-labelled context."""
    passages = "\n\n".join(
        f"[{i}] (source: {h.chunk.source}, page {h.chunk.page})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}"


# --------------------------------------------------------------------------- extractive

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "what", "which", "who",
    "how", "why", "when", "where", "does", "do", "did", "of", "in", "on", "to", "for",
    "and", "or", "it", "its", "this", "that", "with", "as", "by", "can", "from", "at",
    "about", "there", "their", "them", "than", "then", "so", "if", "not", "no",
}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


class ExtractiveLLM:
    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, hits: list[Hit]) -> str:
        wanted = _terms(question)
        scored: list[tuple[float, int, str]] = []
        for hit in hits:
            for sentence in _SENTENCE.split(hit.chunk.text):
                overlap = len(wanted & _terms(sentence))
                if overlap:
                    scored.append((overlap + 0.1 * hit.score, hit.rank, sentence.strip()))
        if not scored:
            first = _SENTENCE.split(hits[0].chunk.text)[0].strip()
            return f"{first} [1]"
        scored.sort(key=lambda item: (-item[0], item[1]))
        picked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _, rank, sentence in scored:
            if sentence in seen:  # overlapping chunks repeat sentences
                continue
            seen.add(sentence)
            picked.append((rank, sentence))
            if len(picked) == self.max_sentences:
                break
        return " ".join(f"{sentence} [{rank}]" for rank, sentence in picked)


# --------------------------------------------------------------------------- Claude


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 600):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Claude backend needs the SDK: pip install anthropic") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (or paste a key in the app sidebar).")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("DOCBOT_ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def answer(self, question: str, hits: list[Hit]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


# --------------------------------------------------------------------------- Ollama


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 120):
        self.model = model or os.environ.get("DOCBOT_OLLAMA_MODEL", "llama3.2")
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = f"http://{host}"
        self.host = host.rstrip("/")
        self.timeout = timeout

    def answer(self, question: str, hits: list[Hit]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, hits)},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} (is `ollama serve` running "
                f"and is the model '{self.model}' pulled?): {exc}"
            ) from exc


LLMS = {"extractive": ExtractiveLLM, "anthropic": AnthropicLLM, "ollama": OllamaLLM}


def get_llm(name: str, **kwargs):
    try:
        factory = LLMS[name]
    except KeyError:
        raise ValueError(f"Unknown LLM '{name}'. Choose from: {', '.join(LLMS)}") from None
    return factory(**kwargs)
