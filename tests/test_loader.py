import pytest

from docbot.loader import UnsupportedFileType, list_documents, load_path, pages_from_bytes


def test_text_and_markdown_become_a_single_page():
    assert pages_from_bytes("a.txt", b"hello") == [(1, "hello")]
    assert pages_from_bytes("README.MD", b"# title") == [(1, "# title")]


def test_invalid_utf8_does_not_crash():
    pages = pages_from_bytes("a.txt", b"caf\xe9 ok")
    assert "ok" in pages[0][1]


def test_pdf_text_is_extracted(make_pdf):
    pages = pages_from_bytes("paper.pdf", make_pdf("Hello PDF world"))
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert "Hello PDF world" in pages[0][1]


def test_unsupported_type_is_rejected():
    with pytest.raises(UnsupportedFileType, match=r"\.docx"):
        pages_from_bytes("report.docx", b"...")


def test_list_documents_filters_and_sorts(tmp_path):
    for name in ["b.txt", "a.md", "c.png", "d.pdf"]:
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_documents(tmp_path)] == ["a.md", "b.txt", "d.pdf"]


def test_list_documents_missing_folder():
    with pytest.raises(FileNotFoundError):
        list_documents("definitely/not/here")


def test_load_path_reads_a_file(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("some notes", encoding="utf-8")
    assert load_path(path) == [(1, "some notes")]
