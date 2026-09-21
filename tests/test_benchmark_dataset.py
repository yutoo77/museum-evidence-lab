from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.benchmark.dataset import BenchmarkDatasetError, load_benchmark_cases
from src.benchmark.models import BenchmarkCategory, BenchmarkSplit
from src.demo_data import DEMO_DOCUMENT_FILENAMES

DATASET = Path("benchmarks/rag_v1/cases.jsonl")


def _valid_item(case_id: str = "case-1") -> dict:
    return {
        "schema_version": 1,
        "id": case_id,
        "split": "smoke",
        "category": "answerable",
        "question": "月が光って見えるのはなぜですか",
        "options": {
            "visitor_profile": "小学生",
            "explanation_scene": "展示前で短く説明",
            "response_language": "やさしい日本語",
        },
        "expected": {
            "status": "answered",
            "answerability": "YES",
            "gold_sources": [
                {"source_name": "moon_phase.txt", "page_number": None}
            ],
            "required_facts": [],
            "forbidden_claims": [],
        },
        "tags": ["unit-test"],
    }


def test_public_dataset_is_nine_smoke_cases_with_balanced_categories():
    cases = load_benchmark_cases(DATASET)

    assert len(cases) == 9
    assert {case.split for case in cases} == {BenchmarkSplit.SMOKE}
    assert Counter(case.category for case in cases) == {
        BenchmarkCategory.ANSWERABLE: 3,
        BenchmarkCategory.UNRELATED: 3,
        BenchmarkCategory.INSUFFICIENT: 3,
    }
    assert len({case.case_id for case in cases}) == len(cases)


def test_public_dataset_gold_sources_are_only_the_five_demo_documents():
    cases = load_benchmark_cases(DATASET)
    sources = {
        source.source_name
        for case in cases
        for source in case.expected.gold_sources
    }

    assert sources <= set(DEMO_DOCUMENT_FILENAMES)
    assert sources == {
        "moon_phase.txt",
        "solar_system.txt",
        "black_hole.txt",
        "planetarium_guide.txt",
        "museum_faq.txt",
    }


def test_loader_rejects_duplicate_ids_with_line_number(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    item = _valid_item()
    path.write_text(
        json.dumps(item, ensure_ascii=False)
        + "\n"
        + json.dumps(item, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDatasetError, match=r":2: idが重複"):
        load_benchmark_cases(path)


def test_loader_rejects_category_expectation_mismatch(tmp_path):
    path = tmp_path / "invalid.jsonl"
    item = _valid_item()
    item["expected"]["status"] = "low_relevance"
    path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(BenchmarkDatasetError, match="分類規則と一致"):
        load_benchmark_cases(path)


def test_loader_requires_gold_source_for_answerable_case(tmp_path):
    path = tmp_path / "missing-source.jsonl"
    item = _valid_item()
    item["expected"]["gold_sources"] = []
    path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(BenchmarkDatasetError, match="1件以上必要"):
        load_benchmark_cases(path)
