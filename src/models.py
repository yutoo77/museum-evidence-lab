"""アプリケーション内部で共有するデータ構造。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PageText:
    """資料のページ単位テキスト。ページなしの形式ではNone。"""

    text: str
    page_number: int | None


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    source_name: str
    file_type: str
    pages: tuple[PageText, ...]
    page_count: int
    warning: str = ""

    @property
    def has_text(self) -> bool:
        return any(page.text.strip() for page in self.pages)


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    page_number: int | None
    chunk_number: int


class RegistrationStatus(StrEnum):
    REGISTERED = "registered"
    REPLACED = "replaced"
    ALREADY_REGISTERED = "already_registered"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    version_id: str
    logical_name: str
    source_name: str
    file_type: str
    content_hash: str
    page_count: int
    chunk_count: int
    registered_at: str
    status: str
    original_path: Path | None
    warning: str = ""
    error_detail: str = ""


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    status: RegistrationStatus
    record: DocumentRecord
    message: str


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    LOW_RELEVANCE = "low_relevance"
    UNANSWERABLE = "unanswerable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SearchHit:
    text: str
    distance: float
    source_name: str
    page_number: int | None
    chunk_number: int
    document_id: str
    version_id: str


@dataclass(frozen=True, slots=True)
class QuestionOptions:
    visitor_profile: str
    explanation_scene: str
    response_language: str
    top_k: int
    distance_threshold: float


@dataclass(frozen=True, slots=True)
class AnswerResult:
    interaction_id: str
    timestamp: str
    question: str
    answer: str
    status: AnswerStatus
    evidence: tuple[SearchHit, ...]
    min_distance: float | None
    distance_threshold: float
    answerability: str
    visitor_profile: str
    explanation_scene: str
    response_language: str
    embedding_model: str
    generation_model: str
    elapsed_seconds: float
    error_detail: str = ""
