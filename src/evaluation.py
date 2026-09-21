"""評価質問セットを同じ質問応答パイプラインで一括実行する。"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from src.models import AnswerStatus, QuestionOptions
from src.qa_service import QuestionAnsweringService

EXPECTED_STATUS = {
    "answerable": AnswerStatus.ANSWERED.value,
    "unrelated": AnswerStatus.LOW_RELEVANCE.value,
    "insufficient": AnswerStatus.UNANSWERABLE.value,
}


def load_evaluation_questions(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("評価質問ファイルの最上位は配列である必要があります。")
    questions: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("評価質問の各項目はオブジェクトである必要があります。")
        category = str(item.get("category", ""))
        if category not in EXPECTED_STATUS or not str(item.get("question", "")).strip():
            raise ValueError(f"不正な評価質問があります: {item}")
        questions.append(item)
    return questions


def run_evaluation(
    service: QuestionAnsweringService,
    questions: Sequence[dict[str, Any]],
    *,
    top_k: int,
    distance_threshold: float,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(questions)
    for index, item in enumerate(questions, start=1):
        question = str(item["question"])
        if progress:
            progress(index, total, question)
        options = QuestionOptions(
            visitor_profile=str(item.get("visitor_profile", "一般の大人")),
            explanation_scene=str(item.get("explanation_scene", "質問に詳しく回答")),
            response_language=str(item.get("response_language", "日本語")),
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        try:
            result = service.answer(question, options)
            status = result.status.value
            minimum = result.min_distance
            answerability = result.answerability
            elapsed = result.elapsed_seconds
        except Exception:
            status = AnswerStatus.ERROR.value
            minimum = None
            answerability = "ERROR"
            elapsed = 0.0
        category = str(item["category"])
        distance_decision = "結果なし"
        if minimum is not None:
            distance_decision = "通過" if minimum <= distance_threshold else "停止"
        rows.append(
            {
                "id": str(item.get("id", index)),
                "question": question,
                "expected_category": category,
                "expected_status": EXPECTED_STATUS[category],
                "min_distance": minimum,
                "distance_decision": distance_decision,
                "answerability": answerability,
                "final_status": status,
                "correct": status == EXPECTED_STATUS[category],
                "elapsed_seconds": elapsed,
            }
        )
    return rows


def summarize_evaluation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    elapsed_values = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    by_category: dict[str, dict[str, int]] = {}
    category_totals = Counter(str(row.get("expected_category")) for row in rows)
    category_correct = Counter(
        str(row.get("expected_category")) for row in rows if row.get("correct")
    )
    for category in category_totals:
        by_category[category] = {
            "correct": category_correct[category],
            "total": category_totals[category],
        }
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "average_seconds": sum(elapsed_values) / total if total else 0.0,
        "by_category": by_category,
        "errors": [row for row in rows if not row.get("correct")],
    }


def evaluation_csv(rows: Sequence[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + output.getvalue()).encode("utf-8")
