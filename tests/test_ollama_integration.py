from __future__ import annotations

import os

import pytest

from src.database import DocumentRepository
from src.document_loader import DocumentLoader
from src.document_service import DocumentRegistrationService
from src.embeddings import OllamaEmbeddingClient
from src.models import RegistrationStatus
from src.vector_store import ChromaVectorStore

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_INTEGRATION") != "1",
    reason="ローカルOllamaを使う明示的な結合テストです。",
)


def test_real_ollama_embedding_can_be_stored_in_chroma(test_config):
    embedding_client = OllamaEmbeddingClient(
        base_url=test_config.ollama.base_url,
        model_name=test_config.ollama.embedding_model,
        timeout_seconds=120,
        batch_size=4,
    )
    assert embedding_client.model_is_available()
    vector_store = ChromaVectorStore(
        test_config.storage.chroma_dir,
        test_config.storage.collection_name,
    )
    service = DocumentRegistrationService(
        config=test_config,
        repository=DocumentRepository(test_config.storage.database_path),
        loader=DocumentLoader(),
        embedding_provider=embedding_client,
        vector_store=vector_store,
    )

    result = service.register_bytes(
        "光の展示.md",
        "# 光\n\n光は電磁波の一種で、人の目に見える範囲を可視光と呼びます。".encode(
            "utf-8"
        ),
    )

    assert result.status == RegistrationStatus.REGISTERED
    assert vector_store.count() == result.record.chunk_count
