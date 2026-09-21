"""ChromaDBへの登録処理を隠蔽するアダプター。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from src.exceptions import VectorStoreError
from src.models import SearchHit, TextChunk


class VectorStore(Protocol):
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
    ) -> None: ...

    def delete_version(self, version_id: str) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int,
        active_version_ids: Sequence[str],
    ) -> list[SearchHit]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    def __init__(self, persist_directory: str | Path, collection_name: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(Path(persist_directory)),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={
                    "description": "科学館職員向け登録資料",
                    "embedding_source": "local_ollama",
                },
            )
        except Exception as exc:
            raise VectorStoreError(
                "登録資料データベースを開けませんでした。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

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
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "資料の検索データを保存できませんでした。",
                "Chunk and embedding counts differ.",
            )
        ids = [f"{document_id}:{version_id}:{chunk.chunk_number}" for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata: dict[str, str | int] = {
                "document_id": document_id,
                "document_version": version_id,
                "source_name": source_name,
                "file_type": file_type,
                "content_hash": content_hash,
                "registered_at": registered_at,
                "chunk_number": chunk.chunk_number,
            }
            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number
            metadatas.append(metadata)
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=[list(vector) for vector in embeddings],
                documents=[chunk.text for chunk in chunks],
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(
                "資料の検索データを保存できませんでした。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def delete_version(self, version_id: str) -> None:
        try:
            self._collection.delete(where={"document_version": version_id})
        except Exception as exc:
            raise VectorStoreError(
                "古い検索データを整理できませんでした。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:
            raise VectorStoreError(
                "登録資料データベースの件数を確認できませんでした。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def search(
        self,
        query_embedding: Sequence[float],
        n_results: int,
        active_version_ids: Sequence[str],
    ) -> list[SearchHit]:
        if not active_version_ids or n_results < 1:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(n_results, max(1, self.count())),
                where={"document_version": {"$in": list(active_version_ids)}},
                include=["documents", "metadatas", "distances"],
            )
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            hits: list[SearchHit] = []
            for document, metadata, distance in zip(
                documents, metadatas, distances, strict=False
            ):
                if document is None or metadata is None or distance is None:
                    continue
                page_value = metadata.get("page_number")
                hits.append(
                    SearchHit(
                        text=str(document),
                        distance=float(distance),
                        source_name=str(metadata.get("source_name", "不明な資料")),
                        page_number=int(page_value) if page_value is not None else None,
                        chunk_number=int(metadata.get("chunk_number", 0)),
                        document_id=str(metadata.get("document_id", "")),
                        version_id=str(metadata.get("document_version", "")),
                    )
                )
            return hits
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "登録資料の検索中にエラーが発生しました。",
                f"{type(exc).__name__}: {exc}",
            ) from exc
