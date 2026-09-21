from __future__ import annotations

import pytest

from src.models import TextChunk
from src.vector_store import ChromaVectorStore


def test_chroma_persists_and_deletes_a_document_version(tmp_path):
    pytest.importorskip(
        "chromadb", reason="Optional old-app test: install requirements-legacy.txt"
    )
    store = ChromaVectorStore(tmp_path / "chroma", "phase1_test_documents")
    chunks = (
        TextChunk("光は電磁波の一種です。", 1, 1),
        TextChunk("可視光は人の目で見える光です。", 2, 2),
    )

    store.upsert_version(
        document_id="doc-1",
        version_id="version-1",
        source_name="展示資料.pdf",
        file_type="pdf",
        content_hash="abc123",
        registered_at="2026-08-03T00:00:00Z",
        chunks=chunks,
        embeddings=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    assert store.count() == 2
    hits = store.search((1.0, 0.0, 0.0), 2, ["version-1"])
    assert hits[0].source_name == "展示資料.pdf"
    assert hits[0].page_number == 1
    assert hits[0].distance < 0.01
    store.delete_version("version-1")
    assert store.count() == 0
