"""RAGベンチマークで共有する不変データ構造。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.models import AnswerStatus


class BenchmarkCategory(StrEnum):
    ANSWERABLE = "answerable"
    UNRELATED = "unrelated"
    INSUFFICIENT = "insufficient"


class BenchmarkSplit(StrEnum):
    """smokeは動作確認専用で、研究上のtest splitではない。"""

    SMOKE = "smoke"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    visitor_profile: str
    explanation_scene: str
    response_language: str


@dataclass(frozen=True, slots=True)
class GoldSource:
    source_name: str
    page_number: int | None = None

    def matches(self, source: RetrievedSource) -> bool:
        if self.source_name != source.source_name:
            return False
        return self.page_number is None or self.page_number == source.page_number


@dataclass(frozen=True, slots=True)
class BenchmarkExpectation:
    status: AnswerStatus
    answerability: str
    gold_sources: tuple[GoldSource, ...]
    required_facts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    split: BenchmarkSplit
    category: BenchmarkCategory
    question: str
    options: BenchmarkOptions
    expected: BenchmarkExpectation
    tags: tuple[str, ...] = ()
    schema_version: int = 1

    @property
    def expected_relevant(self) -> bool:
        """関連資料を通すべき分類か。insufficientも検索段階では関連あり。"""

        return self.category != BenchmarkCategory.UNRELATED


@dataclass(frozen=True, slots=True)
class RetrievedSource:
    source_name: str
    page_number: int | None
    rank: int
    distance: float


@dataclass(frozen=True, slots=True)
class BenchmarkCaseResult:
    case: BenchmarkCase
    actual_status: AnswerStatus
    actual_answerability: str
    retrieved_sources: tuple[RetrievedSource, ...]
    elapsed_seconds: float
    min_distance: float | None = None
    distance_threshold: float | None = None
    distance_relevant: bool | None = None
    answer: str = ""
    error_detail: str = ""

    @property
    def status_correct(self) -> bool:
        return self.actual_status == self.case.expected.status


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    total: int
    correct: int
    status_accuracy: float | None
    false_answer_rate: float | None
    false_refusal_rate: float | None
    error_rate: float | None
    distance_gate_accuracy: float | None
    distance_false_accept_rate: float | None
    distance_false_reject_rate: float | None
    retrieval_k: int
    retrieval_hit_rate: float | None
    mean_reciprocal_rank: float | None
    answerability_accuracy: float | None
    invalid_answerability_rate: float | None
    median_elapsed_seconds: float | None
    p95_elapsed_seconds: float | None
    category_accuracy: dict[str, float | None]
