"""Smoke tests for the Streamlit UI (skipped when Streamlit is not installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_with_the_sample_notes():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception


def test_app_answers_a_question():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.chat_input[0].set_value("What is thrashing?").run()
    assert not app.exception
    assert len(app.chat_message) == 2  # the question and the answer
