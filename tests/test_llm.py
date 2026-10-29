import json
import sys
import types
import urllib.error
import urllib.request

import pytest

from docbot.chunker import Chunk
from docbot.llm import (
    SYSTEM_PROMPT,
    AnthropicLLM,
    ExtractiveLLM,
    OllamaLLM,
    build_prompt,
    get_llm,
)
from docbot.retriever import Hit


def hit(text: str, rank: int = 1, score: float = 0.5, source: str = "notes.txt", page: int = 1):
    return Hit(Chunk(text, source, page, rank - 1), score, rank)


HITS = [
    hit("Cats purr when content. Dogs bark at strangers.", rank=1, source="pets.txt", page=3),
    hit("The ocean is salty.", rank=2, score=0.2),
]


def test_prompt_numbers_passages_and_labels_sources():
    prompt = build_prompt("Why do cats purr?", HITS)
    assert "[1] (source: pets.txt, page 3)" in prompt
    assert "[2] (source: notes.txt, page 1)" in prompt
    assert prompt.endswith("Question: Why do cats purr?")


def test_extractive_returns_the_matching_sentence_with_citation():
    answer = ExtractiveLLM(max_sentences=1).answer("Why do cats purr?", HITS)
    assert answer == "Cats purr when content. [1]"


def test_extractive_does_not_repeat_sentences_from_overlapping_chunks():
    overlapping = [
        hit("Cats purr when content. Dogs bark.", rank=1),
        hit("Cats purr when content. Birds sing.", rank=2),
    ]
    answer = ExtractiveLLM(max_sentences=3).answer("cats purr", overlapping)
    assert answer.count("Cats purr when content.") == 1


def test_unknown_llm_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM"):
        get_llm("gpt-nine")


def test_anthropic_backend_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=object))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


def test_anthropic_backend_sends_grounded_prompt(monkeypatch):
    """Uses a stand-in SDK, so this checks our request shape, not the live API."""
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            block = types.SimpleNamespace(type="text", text="Cats purr when content. [1]")
            return types.SimpleNamespace(content=[block])

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
    answer = AnthropicLLM(api_key="test-key", model="some-model").answer("Why do cats purr?", HITS)

    assert answer == "Cats purr when content. [1]"
    assert calls["api_key"] == "test-key"
    assert calls["model"] == "some-model"
    assert calls["system"] == SYSTEM_PROMPT
    assert "Why do cats purr?" in calls["messages"][0]["content"]


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self, *args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_backend_posts_to_chat_endpoint(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return FakeResponse({"message": {"content": "Because they are content."}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    answer = OllamaLLM(model="tiny", host="localhost:11434").answer("Why do cats purr?", HITS)

    assert answer == "Because they are content."
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["model"] == "tiny"
    assert seen["body"]["stream"] is False


def test_ollama_unreachable_gives_a_helpful_error(monkeypatch):
    def refuse(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="Ollama"):
        OllamaLLM().answer("anything", HITS)
