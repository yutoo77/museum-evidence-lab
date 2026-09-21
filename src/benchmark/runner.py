"""既存QAサービスまたはFakeを同じ形で実行するBenchmark runner。"""

from __future__ import annotations

import time
from typing import Protocol

from src.benchmark.models import BenchmarkCase, BenchmarkCaseResult, RetrievedSource
from src.models import AnswerResult, AnswerStatus, QuestionOptions


class BenchmarkQAService(Protocol):
    def answer(self, question: str, options: QuestionOptions) -> AnswerResult: ...


def run_benchmark(
    service: BenchmarkQAService,
    cases: tuple[BenchmarkCase, ...] | list[BenchmarkCase],
    *,
    top_k: int,
    distance_threshold: float,
) -> tuple[BenchmarkCaseResult, ...]:
    """1件の例外で全体を止めず、後続caseを必ず実行する。"""

    if top_k < 1:
        raise ValueError("top_kは1以上である必要があります。")
    results: list[BenchmarkCaseResult] = []
    for case in cases:
        started = time.perf_counter()
        options = QuestionOptions(
            visitor_profile=case.options.visitor_profile,
            explanation_scene=case.options.explanation_scene,
            response_language=case.options.response_language,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        try:
            answer = service.answer(case.question, options)
            distance_relevant = (
                None
                if answer.status == AnswerStatus.ERROR and answer.min_distance is None
                else answer.min_distance is not None
                and answer.min_distance <= answer.distance_threshold
            )
            retrieved = tuple(
                RetrievedSource(
                    source_name=hit.source_name,
                    page_number=hit.page_number,
                    rank=rank,
                    distance=hit.distance,
                )
                for rank, hit in enumerate(answer.evidence, start=1)
            )
            results.append(
                BenchmarkCaseResult(
                    case=case,
                    actual_status=answer.status,
                    actual_answerability=answer.answerability,
                    retrieved_sources=retrieved,
                    elapsed_seconds=answer.elapsed_seconds,
                    min_distance=answer.min_distance,
                    distance_threshold=answer.distance_threshold,
                    distance_relevant=distance_relevant,
                    answer=answer.answer,
                    error_detail=answer.error_detail,
                )
            )
        except Exception as exc:
            results.append(
                BenchmarkCaseResult(
                    case=case,
                    actual_status=AnswerStatus.ERROR,
                    actual_answerability="ERROR",
                    retrieved_sources=(),
                    elapsed_seconds=round(time.perf_counter() - started, 6),
                    error_detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(results)
