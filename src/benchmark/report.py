"""Benchmark結果を再解析可能なJSONL・JSON・CSVへ保存する。"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.benchmark.metrics import confusion_matrix
from src.benchmark.models import BenchmarkCaseResult, BenchmarkMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    run_manifest: Path
    results_jsonl: Path
    summary_json: Path
    report_csv: Path


def _source_record(source: Any) -> dict[str, Any]:
    return {
        "source_name": source.source_name,
        "page_number": source.page_number,
        "rank": source.rank,
        "distance": source.distance,
    }


def result_record(result: BenchmarkCaseResult) -> dict[str, Any]:
    """根拠本文を含めず、Caseと判定結果を構造化する。"""

    return {
        "schema_version": 1,
        "id": result.case.case_id,
        "split": result.case.split.value,
        "category": result.case.category.value,
        "question": result.case.question,
        "expected": {
            "status": result.case.expected.status.value,
            "answerability": result.case.expected.answerability,
            "gold_sources": [
                {
                    "source_name": source.source_name,
                    "page_number": source.page_number,
                }
                for source in result.case.expected.gold_sources
            ],
        },
        "actual": {
            "status": result.actual_status.value,
            "answerability": result.actual_answerability,
            "status_correct": result.status_correct,
            "min_distance": result.min_distance,
            "distance_threshold": result.distance_threshold,
            "distance_relevant": result.distance_relevant,
            "retrieved_sources": [
                _source_record(source) for source in result.retrieved_sources
            ],
            "elapsed_seconds": result.elapsed_seconds,
            "answer": result.answer,
            "error_detail": result.error_detail,
        },
    }


def _csv_bytes(results: tuple[BenchmarkCaseResult, ...]) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "id",
        "split",
        "category",
        "question",
        "expected_status",
        "actual_status",
        "status_correct",
        "expected_answerability",
        "actual_answerability",
        "min_distance",
        "distance_threshold",
        "distance_relevant",
        "retrieved_sources",
        "elapsed_seconds",
        "answer",
        "error_detail",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "id": result.case.case_id,
                "split": result.case.split.value,
                "category": result.case.category.value,
                "question": result.case.question,
                "expected_status": result.case.expected.status.value,
                "actual_status": result.actual_status.value,
                "status_correct": result.status_correct,
                "expected_answerability": result.case.expected.answerability,
                "actual_answerability": result.actual_answerability,
                "min_distance": result.min_distance,
                "distance_threshold": result.distance_threshold,
                "distance_relevant": result.distance_relevant,
                "retrieved_sources": json.dumps(
                    [_source_record(source) for source in result.retrieved_sources],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "elapsed_seconds": result.elapsed_seconds,
                "answer": result.answer,
                "error_detail": result.error_detail,
            }
        )
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def write_benchmark_artifacts(
    run_directory: str | Path,
    results: tuple[BenchmarkCaseResult, ...],
    metrics: BenchmarkMetrics,
    run_manifest: dict[str, Any],
) -> BenchmarkArtifacts:
    root = Path(run_directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    results_path = root / "results.jsonl"
    summary_path = root / "summary.json"
    csv_path = root / "report.csv"

    manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    records = [result_record(result) for result in results]
    results_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "metrics": asdict(metrics),
        "confusion_matrix": confusion_matrix(results),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path.write_bytes(_csv_bytes(results))
    return BenchmarkArtifacts(manifest_path, results_path, summary_path, csv_path)
