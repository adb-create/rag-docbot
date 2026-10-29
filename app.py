"""Streamlit chat UI for DocBot.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from docbot import RAGPipeline
from docbot.llm import get_llm
from docbot.loader import list_documents, load_path, pages_from_bytes

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_docs"
LLM_CHOICES = {
    "Extractive (no API key needed)": "extractive",
    "Claude (Anthropic API)": "anthropic",
    "Ollama (local model)": "ollama",
}
EMBEDDER_CHOICES = {
    "TF-IDF (fast, offline)": "tfidf",
    "Sentence-Transformers (semantic)": "sentence-transformers",
}

st.set_page_config(page_title="DocBot: chat with your documents", layout="wide")


@st.cache_data(show_spinner=False)
def parse_upload(name: str, data: bytes):
    return pages_from_bytes(name, data)


@st.cache_data(show_spinner=False)
def parse_sample(path: str):
    return load_path(path)


def collect_documents(use_samples: bool, uploads) -> tuple[dict, list[str]]:
    """Return ({file name: pages}, [problems]) for the sample notes and uploaded files."""
    docs: dict = {}
    problems: list[str] = []
    if use_samples:
        for path in list_documents(SAMPLE_DIR):
            docs[path.name] = parse_sample(str(path))
    for uploaded in uploads or []:
        try:
            docs[uploaded.name] = parse_upload(uploaded.name, uploaded.getvalue())
        except Exception as exc:  # a bad file should not take the whole app down
            problems.append(f"{uploaded.name}: {exc}")
    return docs, problems


def build_pipeline(docs: dict, embedder: str, chunk_size: int, overlap: int) -> RAGPipeline:
    pipeline = RAGPipeline(
        embedder=embedder, llm="extractive", chunk_size=chunk_size, overlap=overlap
    )
    for name, pages in docs.items():
        pipeline.add_pages(name, pages)
    pipeline.build()
    return pipeline


def render_sources(hits, abstained: bool) -> None:
    label = (
        "Closest passages (below the relevance threshold, so not used)"
        if abstained
        else f"Sources ({len(hits)} passage{'' if len(hits) == 1 else 's'})"
    )
    with st.expander(label):
        for hit in hits:
            st.markdown(
                f"**[{hit.rank}] {hit.chunk.source}, page {hit.chunk.page}**"
                f" · relevance {hit.score:.2f}"
            )
            st.caption(hit.chunk.text)


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Documents")
    use_samples = st.toggle("Use the sample study notes", value=True)
    uploads = st.file_uploader(
        "Upload PDF, TXT or Markdown files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )

    st.header("Answering")
    llm_label = st.selectbox("Answer engine", list(LLM_CHOICES))
    llm_name = LLM_CHOICES[llm_label]
    api_key = ""
    if llm_name == "anthropic":
        api_key = st.text_input(
            "Anthropic API key",
            type="password",
            help="Leave empty to use the ANTHROPIC_API_KEY environment variable.",
        )

    st.header("Retrieval")
    embedder_label = st.selectbox("Embeddings", list(EMBEDDER_CHOICES))
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 600, step=100)
    overlap_pct = st.slider("Chunk overlap (%)", 0, 40, 15, step=5)
    top_k = st.slider("Passages to retrieve", 1, 8, 4)

    if st.button("Clear chat"):
        st.session_state.messages = []

# --------------------------------------------------------------------------- main

st.title("DocBot")
st.caption(
    "Ask questions about your documents. Answers are built only from retrieved passages, "
    "and every answer shows its sources."
)

docs, problems = collect_documents(use_samples, uploads)
for problem in problems:
    st.warning(f"Skipped {problem}")
if not docs:
    st.info("Upload a document or switch on the sample study notes in the sidebar to begin.")
    st.stop()

overlap = int(chunk_size * overlap_pct / 100)
signature = (
    EMBEDDER_CHOICES[embedder_label],
    chunk_size,
    overlap,
    tuple(sorted((name, sum(len(text) for _, text in pages)) for name, pages in docs.items())),
)
if st.session_state.get("signature") != signature:
    try:
        with st.spinner("Indexing documents..."):
            st.session_state.pipeline = build_pipeline(
                docs, EMBEDDER_CHOICES[embedder_label], chunk_size, overlap
            )
        st.session_state.signature = signature
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

pipeline = st.session_state.pipeline
st.sidebar.caption(f"{len(pipeline.sources)} documents · {pipeline.n_chunks} chunks indexed")

if "messages" not in st.session_state:
    st.session_state.messages = []
messages = st.session_state.messages

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("hits"):
            render_sources(message["hits"], message.get("abstained", False))

question = st.chat_input("Ask something about your documents")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm_options = {"api_key": api_key} if llm_name == "anthropic" and api_key else {}
            pipeline.llm = get_llm(llm_name, **llm_options)
            pipeline.top_k = top_k
            with st.spinner("Searching your documents..."):
                answer = pipeline.ask(question)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
            st.stop()
        st.markdown(answer.text)
        render_sources(answer.hits, answer.abstained)

    messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "hits": answer.hits,
            "abstained": answer.abstained,
        }
    )
