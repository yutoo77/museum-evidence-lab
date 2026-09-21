from __future__ import annotations

import pytest

from src.config import (
    AppConfig,
    AppSection,
    ChunkingSection,
    OllamaSection,
    RetrievalSection,
    StorageSection,
)


@pytest.fixture
def test_config(tmp_path):
    config = AppConfig(
        project_root=tmp_path,
        app=AppSection(title="test", max_upload_mb=5),
        ollama=OllamaSection(
            base_url="http://127.0.0.1:11434",
            embedding_model="embeddinggemma",
            generation_model="qwen3:1.7b",
            timeout_seconds=2,
            embedding_batch_size=4,
        ),
        chunking=ChunkingSection(chunk_size=120, chunk_overlap=20),
        retrieval=RetrievalSection(top_k=5, distance_threshold=0.45),
        storage=StorageSection(
            documents_dir=tmp_path / "documents",
            chroma_dir=tmp_path / "chroma",
            database_path=tmp_path / "app.db",
            log_dir=tmp_path / "logs",
            interaction_log_path=tmp_path / "logs" / "interactions.jsonl",
            collection_name="test_documents",
        ),
    )
    config.ensure_directories()
    return config
