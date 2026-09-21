"""JSONLベンチマークデータを厳格に読み込む。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.benchmark.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkExpectation,
    BenchmarkOptions,
    BenchmarkSplit,
    GoldSource,
)
from src.models import AnswerStatus


class BenchmarkDatasetError(ValueError):
    """データセットの構造または意味制約が不正。"""


_EXPECTED_BY_CATEGORY = {
    BenchmarkCategory.ANSWERABLE: (AnswerStatus.ANSWERED, "YES", True),
    BenchmarkCategory.UNRELATED: (AnswerStatus.LOW_RELEVANCE, "NOT_RUN", False),
    BenchmarkCategory.INSUFFICIENT: (AnswerStatus.UNANSWERABLE, "NO", True),
}


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkDatasetError(f"{field}は空でない文字列である必要があります。")
    return value.strip()


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkDatasetError(f"{field}はオブジェクトである必要があります。")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BenchmarkDatasetError(f"{field}は文字列配列である必要があります。")
    return tuple(_nonempty_string(item, f"{field}[]") for item in value)


def _parse_gold_sources(value: Any) -> tuple[GoldSource, ...]:
    if not isinstance(value, list):
        raise BenchmarkDatasetError("expected.gold_sourcesは配列である必要があります。")
    sources: list[GoldSource] = []
    for index, item in enumerate(value):
        source = _object(item, f"expected.gold_sources[{index}]")
        page = source.get("page_number")
        if page is not None and (
            not isinstance(page, int) or isinstance(page, bool) or page < 1
        ):
            raise BenchmarkDatasetError(
                f"expected.gold_sources[{index}].page_numberは1以上の整数またはnullです。"
            )
        sources.append(
            GoldSource(
                source_name=_nonempty_string(
                    source.get("source_name"),
                    f"expected.gold_sources[{index}].source_name",
                ),
                page_number=page,
            )
        )
    return tuple(sources)


def _parse_case(item: Any) -> BenchmarkCase:
    data = _object(item, "case")
    if data.get("schema_version") != 1:
        raise BenchmarkDatasetError("schema_versionは1である必要があります。")
    try:
        split = BenchmarkSplit(data.get("split"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkDatasetError("splitはsmoke、dev、testのいずれかです。") from exc
    try:
        category = BenchmarkCategory(data.get("category"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkDatasetError(
            "categoryはanswerable、unrelated、insufficientのいずれかです。"
        ) from exc

    options_data = _object(data.get("options"), "options")
    options = BenchmarkOptions(
        visitor_profile=_nonempty_string(
            options_data.get("visitor_profile"), "options.visitor_profile"
        ),
        explanation_scene=_nonempty_string(
            options_data.get("explanation_scene"), "options.explanation_scene"
        ),
        response_language=_nonempty_string(
            options_data.get("response_language"), "options.response_language"
        ),
    )

    expected_data = _object(data.get("expected"), "expected")
    try:
        status = AnswerStatus(expected_data.get("status"))
    except (TypeError, ValueError) as exc:
        raise BenchmarkDatasetError("expected.statusが不正です。") from exc
    answerability = _nonempty_string(
        expected_data.get("answerability"), "expected.answerability"
    )
    if answerability not in {"YES", "NO", "NOT_RUN"}:
        raise BenchmarkDatasetError(
            "expected.answerabilityはYES、NO、NOT_RUNのいずれかです。"
        )
    gold_sources = _parse_gold_sources(expected_data.get("gold_sources"))
    expected_status, expected_gate, needs_source = _EXPECTED_BY_CATEGORY[category]
    if status != expected_status or answerability != expected_gate:
        raise BenchmarkDatasetError(
            f"{category.value}の期待status/answerabilityが分類規則と一致しません。"
        )
    if needs_source != bool(gold_sources):
        requirement = "1件以上必要" if needs_source else "空配列である必要"
        raise BenchmarkDatasetError(
            f"{category.value}のexpected.gold_sourcesは{requirement}があります。"
        )

    return BenchmarkCase(
        case_id=_nonempty_string(data.get("id"), "id"),
        split=split,
        category=category,
        question=_nonempty_string(data.get("question"), "question"),
        options=options,
        expected=BenchmarkExpectation(
            status=status,
            answerability=answerability,
            gold_sources=gold_sources,
            required_facts=_string_tuple(
                expected_data.get("required_facts", []), "expected.required_facts"
            ),
            forbidden_claims=_string_tuple(
                expected_data.get("forbidden_claims", []),
                "expected.forbidden_claims",
            ),
        ),
        tags=_string_tuple(data.get("tags", []), "tags"),
    )


def load_benchmark_cases(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """JSONLを読み込み、行番号つきの安全なエラーへ変換する。"""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BenchmarkDatasetError(f"データセットを読み込めません: {source}") from exc

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case = _parse_case(item)
        except (json.JSONDecodeError, BenchmarkDatasetError) as exc:
            raise BenchmarkDatasetError(f"{source}:{line_number}: {exc}") from exc
        if case.case_id in seen_ids:
            raise BenchmarkDatasetError(
                f"{source}:{line_number}: idが重複しています: {case.case_id}"
            )
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise BenchmarkDatasetError(f"データセットが空です: {source}")
    return tuple(cases)
