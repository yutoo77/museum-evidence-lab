from __future__ import annotations

from collections.abc import Sequence

from src.database import DocumentRepository
from src.document_loader import DocumentLoader
from src.document_service import DocumentRegistrationService
from src.models import RegistrationStatus, TextChunk


class FakeEmbeddingProvider:
    model_name = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(index), 1.0, 0.5] for index, _ in enumerate(texts, start=1)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.versions: dict[str, list[str]] = {}
        self.upsert_calls = 0

    def upsert_version(
        self,
        *,
        document_id: str,
        version_id: str,
        source_name: str,
        file_type: str,
        content_hash: str,
        registered_at: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.upsert_calls += 1
        assert len(chunks) == len(embeddings)
        self.versions[version_id] = [chunk.text for chunk in chunks]

    def delete_version(self, version_id: str) -> None:
        self.versions.pop(version_id, None)


def build_service(test_config):
    embeddings = FakeEmbeddingProvider()
    vectors = FakeVectorStore()
    repository = DocumentRepository(test_config.storage.database_path)
    service = DocumentRegistrationService(
        config=test_config,
        repository=repository,
        loader=DocumentLoader(),
        embedding_provider=embeddings,
        vector_store=vectors,
    )
    return service, repository, embeddings, vectors


def test_same_document_is_not_registered_twice(test_config):
    service, repository, embeddings, vectors = build_service(test_config)
    content = "光は電磁波の一種です。".encode("utf-8")

    first = service.register_bytes("展示資料.txt", content)
    second = service.register_bytes("展示資料.txt", content)

    assert first.status == RegistrationStatus.REGISTERED
    assert second.status == RegistrationStatus.ALREADY_REGISTERED
    assert len(repository.list_documents()) == 1
    assert embeddings.calls == 1
    assert vectors.upsert_calls == 1
    assert len(vectors.versions) == 1


def test_same_name_with_new_content_replaces_old_version(test_config):
    service, repository, _, vectors = build_service(test_config)

    first = service.register_bytes("展示資料.txt", "古い説明です。".encode("utf-8"))
    second = service.register_bytes("展示資料.txt", "新しい説明です。".encode("utf-8"))

    records = repository.list_documents()
    assert second.status == RegistrationStatus.REPLACED
    assert first.record.document_id == second.record.document_id
    assert len(records) == 1
    assert records[0].content_hash == second.record.content_hash
    assert list(vectors.versions) == [second.record.version_id]
