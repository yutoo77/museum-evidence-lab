"""外部I/Oを持たないRAGベンチマーク指標。"""

from __future__ import annotations

from collections.abc import Sequence

from src.benchmark.models import BenchmarkCaseResult, BenchmarkMetrics
from src.models import AnswerStatus


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def status_accuracy(results: Sequence[BenchmarkCaseResult]) -> float | None:
    return _rate(sum(result.status_correct for result in results), len(results))


def false_answer_rate(results: Sequence[BenchmarkCaseResult]) -> float | None:
    nonanswerable = [
        result
        for result in results
        if result.case.expected.status != AnswerStatus.ANSWERED
    ]
    return _rate(
        sum(result.actual_status == AnswerStatus.ANSWERED for result in nonanswerable),
        len(nonanswerable),
    )


def false_refusal_rate(results: Sequence[BenchmarkCaseResult]) -> float | None:
    answerable = [
        result
        for result in results
        if result.case.expected.status == AnswerStatus.ANSWERED
    ]
    return _rate(
        sum(result.actual_status != AnswerStatus.ANSWERED for result in answerable),
        len(answerable),
    )


def _distance_results(
    results: Sequence[BenchmarkCaseResult],
) -> list[BenchmarkCaseResult]:
    return [result for result in results if result.distance_relevant is not None]


def distance_gate_accuracy(results: Sequence[BenchmarkCaseResult]) -> float | None:
    evaluated = _distance_results(results)
    return _rate(
        sum(
            result.distance_relevant == result.case.expected_relevant
            for result in evaluated
        ),
        len(evaluated),
    )


def distance_false_accept_rate(
    results: Sequence[BenchmarkCaseResult],
) -> float | None:
    expected_irrelevant = [
        result
        for result in _distance_results(results)
        if not result.case.expected_relevant
    ]
    return _rate(
        sum(result.distance_relevant is True for result in expected_irrelevant),
        len(expected_irrelevant),
    )


def distance_false_reject_rate(
    results: Sequence[BenchmarkCaseResult],
) -> float | None:
    expected_relevant = [
        result
        for result in _distance_results(results)
        if result.case.expected_relevant
    ]
    return _rate(
        sum(result.distance_relevant is False for result in expected_relevant),
        len(expected_relevant),
    )


def _first_gold_rank(result: BenchmarkCaseResult, k: int) -> int | None:
    if k < 1:
        raise ValueError("kは1以上である必要があります。")
    for source in result.retrieved_sources:
        if source.rank > k:
            continue
        if any(gold.matches(source) for gold in result.case.expected.gold_sources):
            return source.rank
    return None


def retrieval_hit_rate_at_k(
    results: Sequence[BenchmarkCaseResult], k: int
) -> float | None:
    with_gold = [result for result in results if result.case.expected.gold_sources]
    return _rate(
        sum(_first_gold_rank(result, k) is not None for result in with_gold),
        len(with_gold),
    )


def mean_reciprocal_rank_at_k(
    results: Sequence[BenchmarkCaseResult], k: int
) -> float | None:
    with_gold = [result for result in results if result.case.expected.gold_sources]
    if not with_gold:
        return None
    reciprocal_ranks = []
    for result in with_gold:
        rank = _first_gold_rank(result, k)
        reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def answerability_accuracy(results: Sequence[BenchmarkCaseResult]) -> float | None:
    gated = [
        result
        for result in results
        if result.case.expected.answerability in {"YES", "NO"}
    ]
    return _rate(
        sum(
            result.actual_answerability == result.case.expected.answerability
            for result in gated
        ),
        len(gated),
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 1:
        raise ValueError("percentileは0以上1以下である必要があります。")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def confusion_matrix(
    results: Sequence[BenchmarkCaseResult],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for result in results:
        expected = result.case.expected.status.value
        actual = result.actual_status.value
        row = matrix.setdefault(expected, {})
        row[actual] = row.get(actual, 0) + 1
    return matrix


def compute_metrics(
    results: Sequence[BenchmarkCaseResult], *, retrieval_k: int = 5
) -> BenchmarkMetrics:
    if retrieval_k < 1:
        raise ValueError("retrieval_kは1以上である必要があります。")
    gated = [
        result
        for result in results
        if result.case.expected.answerability in {"YES", "NO"}
    ]
    categories = sorted({result.case.category.value for result in results})
    category_accuracy = {
        category: status_accuracy(
            [result for result in results if result.case.category.value == category]
        )
        for category in categories
    }
    elapsed = [result.elapsed_seconds for result in results]
    return BenchmarkMetrics(
        total=len(results),
        correct=sum(result.status_correct for result in results),
        status_accuracy=status_accuracy(results),
        false_answer_rate=false_answer_rate(results),
        false_refusal_rate=false_refusal_rate(results),
        error_rate=_rate(
            sum(result.actual_status == AnswerStatus.ERROR for result in results),
            len(results),
        ),
        distance_gate_accuracy=distance_gate_accuracy(results),
        distance_false_accept_rate=distance_false_accept_rate(results),
        distance_false_reject_rate=distance_false_reject_rate(results),
        retrieval_k=retrieval_k,
        retrieval_hit_rate=retrieval_hit_rate_at_k(results, retrieval_k),
        mean_reciprocal_rank=mean_reciprocal_rank_at_k(results, retrieval_k),
        answerability_accuracy=answerability_accuracy(results),
        invalid_answerability_rate=_rate(
            sum(result.actual_answerability == "INVALID" for result in gated),
            len(gated),
        ),
        median_elapsed_seconds=_percentile(elapsed, 0.5),
        p95_elapsed_seconds=_percentile(elapsed, 0.95),
        category_accuracy=category_accuracy,
    )
