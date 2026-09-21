from __future__ import annotations

from pathlib import Path

from src.demo_data import DEMO_DOCUMENT_FILENAMES
from src.evaluation import load_evaluation_questions


def test_all_demo_documents_exist_and_are_marked_fictional():
    for filename in DEMO_DOCUMENT_FILENAMES:
        text = (Path("sample_docs") / filename).read_text(encoding="utf-8")
        assert "架空" in text


def test_evaluation_set_has_at_least_three_questions_per_category():
    questions = load_evaluation_questions("evaluation_questions.json")
    counts = {
        category: sum(item["category"] == category for item in questions)
        for category in ("answerable", "unrelated", "insufficient")
    }

    assert len(questions) >= 9
    assert all(count >= 3 for count in counts.values())
