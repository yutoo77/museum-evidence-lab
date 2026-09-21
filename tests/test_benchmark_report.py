from __future__ import annotations

import json

from src.benchmark.metrics import compute_metrics
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
from src.benchmark.report import result_record, write_benchmark_artifacts
from src.models import AnswerStatus


def _result() -> BenchmarkCaseResult:
    case = BenchmarkCase(
        case_id="moon-1",
        split=BenchmarkSplit.SMOKE,
        category=BenchmarkCategory.ANSWERABLE,
        question="月の満ち欠けとは？",
        options=BenchmarkOptions("小学生", "展示前で短く説明", "やさしい日本語"),
        expected=BenchmarkExpectation(
            AnswerStatus.ANSWERED,
            "YES",
            (GoldSource("moon_phase.txt"),),
        ),
    )
    return BenchmarkCaseResult(
        case=case,
        actual_status=AnswerStatus.ANSWERED,
        actual_answerability="YES",
        retrieved_sources=(RetrievedSource("moon_phase.txt", None, 1, 0.2),),
        elapsed_seconds=1.2,
        min_distance=0.2,
        distance_threshold=0.75,
        distance_relevant=True,
        answer="資料に基づく回答",
    )


def test_result_record_does_not_store_evidence_body():
    record = result_record(_result())

    assert record["actual"]["retrieved_sources"] == [
        {
            "source_name": "moon_phase.txt",
            "page_number": None,
            "rank": 1,
            "distance": 0.2,
        }
    ]
    assert "text" not in record["actual"]["retrieved_sources"][0]


def test_report_writes_machine_readable_artifacts(tmp_path):
    results = (_result(),)
    metrics = compute_metrics(results, retrieval_k=1)

    artifacts = write_benchmark_artifacts(
        tmp_path,
        results,
        metrics,
        {"schema_version": 1, "run_id": "test", "complete": True},
    )

    summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    lines = artifacts.results_jsonl.read_text(encoding="utf-8").splitlines()
    assert summary["metrics"]["status_accuracy"] == 1.0
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "moon-1"
    assert artifacts.report_csv.read_bytes().startswith(b"\xef\xbb\xbf")
