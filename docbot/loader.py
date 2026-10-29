"""Read documents into (page_number, text) pairs."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

Pages = list[tuple[int, str]]


class UnsupportedFileType(ValueError):
    """Raised for files the loader does not know how to read."""


def pages_from_bytes(name: str, data: bytes) -> Pages:
    """Parse an in-memory file (e.g. a Streamlit upload) into pages."""
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many "encrypted" PDFs only have an empty owner password.
            try:
                reader.decrypt("")
            except Exception as exc:  # pragma: no cover - depends on the file
                raise ValueError(f"{name} is password-protected") from exc
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise UnsupportedFileType(
        f"Unsupported file type '{suffix}' for {name}. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def load_path(path: str | Path) -> Pages:
    path = Path(path)
    return pages_from_bytes(path.name, path.read_bytes())


def list_documents(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
