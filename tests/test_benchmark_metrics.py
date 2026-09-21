from __future__ import annotations

import pytest

from src.benchmark.metrics import compute_metrics, confusion_matrix
from src.benchmark.models import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkCategory,
    BenchmarkExpectation,
    BenchmarkOptions,
    BenchmarkSplit,
    GoldSource,
    RetrievedSource,
)
from src.models import AnswerStatus


def _case(
    case_id: str,
    category: BenchmarkCategory,
    expected_status: AnswerStatus,
    expected_answerability: str,
    gold_sources: tuple[GoldSource, ...] = (),
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        split=BenchmarkSplit.SMOKE,
        category=category,
        question=case_id,
        options=BenchmarkOptions("一般の大人", "質問に詳しく回答", "日本語"),
        expected=BenchmarkExpectation(
            expected_status, expected_answerability, gold_sources
        ),
    )


def _result(
    case: BenchmarkCase,
    actual_status: AnswerStatus,
    actual_answerability: str,
    retrieved: tuple[RetrievedSource, ...] = (),
    elapsed: float = 1.0,
    distance_relevant: bool | None = True,
) -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case=case,
        actual_status=actual_status,
        actual_answerability=actual_answerability,
        retrieved_sources=retrieved,
        elapsed_seconds=elapsed,
        min_distance=0.2 if distance_relevant is not None else None,
        distance_threshold=0.75,
        distance_relevant=distance_relevant,
    )


def test_metrics_separate_safety_refusal_retrieval_and_gate_accuracy():
    moon = GoldSource("moon_phase.txt")
    answerable = _case(
        "answerable",
        BenchmarkCategory.ANSWERABLE,
        AnswerStatus.ANSWERED,
        "YES",
        (moon,),
    )
    unrelated = _case(
        "unrelated",
        BenchmarkCategory.UNRELATED,
        AnswerStatus.LOW_RELEVANCE,
        "NOT_RUN",
    )
    insufficient = _case(
        "insufficient",
        BenchmarkCategory.INSUFFICIENT,
        AnswerStatus.UNANSWERABLE,
        "NO",
        (moon,),
    )
    results = (
        _result(
            answerable,
            AnswerStatus.UNANSWERABLE,
            "NO",
            (
                RetrievedSource("other.txt", None, 1, 0.1),
                RetrievedSource("moon_phase.txt", None, 2, 0.2),
            ),
            1.0,
        ),
        _result(
            unrelated,
            AnswerStatus.ANSWERED,
            "NOT_RUN",
            elapsed=2.0,
            distance_relevant=True,
        ),
        _result(
            insufficient,
            AnswerStatus.UNANSWERABLE,
            "NO",
            (RetrievedSource("moon_phase.txt", None, 1, 0.2),),
            3.0,
        ),
    )

    metrics = compute_metrics(results, retrieval_k=2)

    assert metrics.total == 3
    assert metrics.correct == 1
    assert metrics.status_accuracy == pytest.approx(1 / 3)
    assert metrics.false_answer_rate == 0.5
    assert metrics.false_refusal_rate == 1.0
    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)
    assert metrics.answerability_accuracy == 0.5
    assert metrics.distance_gate_accuracy == pytest.approx(2 / 3)
    assert metrics.distance_false_accept_rate == 1.0
    assert metrics.distance_false_reject_rate == 0.0
    assert metrics.median_elapsed_seconds == 2.0
    assert metrics.p95_elapsed_seconds == pytest.approx(2.9)


def test_gold_page_none_matches_any_page_but_specific_page_must_match():
    any_page = GoldSource("guide.pdf", None)
    page_two = GoldSource("guide.pdf", 2)
    source = RetrievedSource("guide.pdf", 3, 1, 0.1)

    assert any_page.matches(source) is True
    assert page_two.matches(source) is False


def test_confusion_matrix_keeps_error_as_an_actual_status():
    case = _case(
        "answerable",
        BenchmarkCategory.ANSWERABLE,
        AnswerStatus.ANSWERED,
        "YES",
        (GoldSource("moon_phase.txt"),),
    )
    matrix = confusion_matrix((_result(case, AnswerStatus.ERROR, "ERROR"),))

    assert matrix == {"answered": {"error": 1}}


def test_empty_results_return_none_rates_instead_of_dividing_by_zero():
    metrics = compute_metrics((), retrieval_k=5)

    assert metrics.total == 0
    assert metrics.status_accuracy is None
    assert metrics.false_answer_rate is None
    assert metrics.false_refusal_rate is None
    assert metrics.retrieval_hit_rate is None
    assert metrics.mean_reciprocal_rank is None
    assert metrics.distance_gate_accuracy is None
    assert metrics.distance_false_accept_rate is None
    assert metrics.distance_false_reject_rate is None


def test_unknown_distance_decision_is_excluded_from_gate_rates():
    case = _case(
        "answerable",
        BenchmarkCategory.ANSWERABLE,
        AnswerStatus.ANSWERED,
        "YES",
        (GoldSource("moon_phase.txt"),),
    )
    metrics = compute_metrics(
        (_result(case, AnswerStatus.ERROR, "ERROR", distance_relevant=None),)
    )

    assert metrics.error_rate == 1.0
    assert metrics.distance_gate_accuracy is None
    assert metrics.distance_false_reject_rate is None
