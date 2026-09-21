from __future__ import annotations

from src.chunking import chunk_document, split_text
from src.models import LoadedDocument, PageText


def test_split_text_respects_size_and_creates_overlap():
    text = "第一文です。" * 50
    chunks = split_text(text, chunk_size=60, overlap=10)

    assert len(chunks) > 1
    assert all(1 <= len(chunk) <= 60 for chunk in chunks)
    assert chunks[0][-5:] in chunks[1]


def test_chunks_never_cross_pdf_pages():
    document = LoadedDocument(
        source_name="sample.pdf",
        file_type="pdf",
        pages=(
            PageText("A" * 150, 1),
            PageText("B" * 150, 2),
        ),
        page_count=2,
    )

    chunks = chunk_document(document, chunk_size=100, overlap=20)

    assert [chunk.chunk_number for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(not ({"A", "B"} <= set(chunk.text)) for chunk in chunks)
