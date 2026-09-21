"""質問のベクトル化と、登録済み版だけを対象にした関連資料検索。"""

from __future__ import annotations

from dataclasses import dataclass

from src.database import DocumentRepository
from src.embeddings import EmbeddingProvider
from src.exceptions import RetrievalError
from src.models import SearchHit
from src.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    hits: tuple[SearchHit, ...]
    min_distance: float | None
    is_relevant: bool


def evaluate_relevance(
    hits: list[SearchHit] | tuple[SearchHit, ...], threshold: float
) -> tuple[float | None, bool]:
    if not hits:
        return None, False
    minimum = min(hit.distance for hit in hits)
    return minimum, minimum <= threshold


class RetrievalService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(self, question: str, top_k: int, threshold: float) -> RetrievalOutcome:
        if not question.strip():
            raise RetrievalError(
                "質問を入力してください。",
                "Question was empty.",
            )
        active_versions = self.repository.active_version_ids()
        if not active_versions:
            return RetrievalOutcome((), None, False)
        vectors = self.embedding_provider.embed_texts([question.strip()])
        if len(vectors) != 1:
            raise RetrievalError(
                "質問を検索用データへ変換できませんでした。",
                f"Expected one query embedding, got {len(vectors)}.",
            )
        hits = self.vector_store.search(vectors[0], top_k, active_versions)
        minimum, relevant = evaluate_relevance(hits, threshold)
        return RetrievalOutcome(tuple(hits), minimum, relevant)
