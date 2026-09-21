from __future__ import annotations

from collections import deque

from src.benchmark.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkExpectation,
    BenchmarkOptions,
    BenchmarkSplit,
    GoldSource,
)
from src.benchmark.runner import run_benchmark
from src.models import AnswerResult, AnswerStatus, SearchHit


class FakeQAService:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def answer(self, question, options):
        self.calls.append((question, options))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


def _case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        split=BenchmarkSplit.SMOKE,
        category=BenchmarkCategory.ANSWERABLE,
        question=f"質問-{case_id}",
        options=BenchmarkOptions("小学生", "展示前で短く説明", "やさしい日本語"),
        expected=BenchmarkExpectation(
            AnswerStatus.ANSWERED, "YES", (GoldSource("moon_phase.txt"),)
        ),
    )


def _answer(case_id: str) -> AnswerResult:
    return AnswerResult(
        interaction_id=case_id,
        timestamp="2026-08-11T00:00:00Z",
        question=f"質問-{case_id}",
        answer="月は太陽の光を反射します。",
        status=AnswerStatus.ANSWERED,
        evidence=(
            SearchHit(
                "月は太陽の光を反射します。",
                0.2,
                "moon_phase.txt",
                None,
                1,
                "doc",
                "version",
            ),
        ),
        min_distance=0.2,
        distance_threshold=0.75,
        answerability="YES",
        visitor_profile="小学生",
        explanation_scene="展示前で短く説明",
        response_language="やさしい日本語",
        embedding_model="fake-embed",
        generation_model="fake-llm",
        elapsed_seconds=1.25,
    )


def test_runner_accepts_fake_qa_and_maps_evidence_rank():
    service = FakeQAService([_answer("one")])

    results = run_benchmark(
        service, (_case("one"),), top_k=3, distance_threshold=0.7
    )

    assert results[0].actual_status == AnswerStatus.ANSWERED
    assert results[0].retrieved_sources[0].source_name == "moon_phase.txt"
    assert results[0].retrieved_sources[0].rank == 1
    assert results[0].elapsed_seconds == 1.25
    assert results[0].min_distance == 0.2
    assert results[0].distance_threshold == 0.75
    assert results[0].distance_relevant is True
    _, options = service.calls[0]
    assert options.top_k == 3
    assert options.distance_threshold == 0.7


def test_one_service_exception_becomes_error_and_does_not_stop_next_case():
    service = FakeQAService([RuntimeError("failed"), _answer("two")])

    results = run_benchmark(
        service,
        (_case("one"), _case("two")),
        top_k=5,
        distance_threshold=0.75,
    )

    assert [result.actual_status for result in results] == [
        AnswerStatus.ERROR,
        AnswerStatus.ANSWERED,
    ]
    assert results[0].actual_answerability == "ERROR"
    assert results[0].distance_relevant is None
    assert "RuntimeError: failed" in results[0].error_detail
    assert len(service.calls) == 2


def test_runner_rejects_invalid_top_k_before_calling_service():
    service = FakeQAService([])

    try:
        run_benchmark(service, (_case("one"),), top_k=0, distance_threshold=0.75)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")
    assert service.calls == []
