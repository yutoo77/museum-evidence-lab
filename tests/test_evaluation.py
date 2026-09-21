from __future__ import annotations

from src.evaluation import evaluation_csv, run_evaluation, summarize_evaluation
from src.models import AnswerResult, AnswerStatus


class FakeQAService:
    def answer(self, question, options):
        if question == "error":
            raise RuntimeError("one question failed")
        status = {
            "answer": AnswerStatus.ANSWERED,
            "unrelated": AnswerStatus.LOW_RELEVANCE,
            "insufficient": AnswerStatus.UNANSWERABLE,
        }[question]
        return AnswerResult(
            interaction_id=question,
            timestamp="2026-08-04T00:00:00Z",
            question=question,
            answer="result",
            status=status,
            evidence=(),
            min_distance=0.2 if status != AnswerStatus.LOW_RELEVANCE else 0.8,
            distance_threshold=options.distance_threshold,
            answerability="YES" if status == AnswerStatus.ANSWERED else "NO",
            visitor_profile=options.visitor_profile,
            explanation_scene=options.explanation_scene,
            response_language=options.response_language,
            embedding_model="embed",
            generation_model="llm",
            elapsed_seconds=1.0,
        )


def question(category, question):
    return {"id": question, "category": category, "question": question}


def test_evaluation_runs_all_classes_and_exports_utf8_csv():
    rows = run_evaluation(
        FakeQAService(),
        [
            question("answerable", "answer"),
            question("unrelated", "unrelated"),
            question("insufficient", "insufficient"),
        ],
        top_k=5,
        distance_threshold=0.45,
    )
    summary = summarize_evaluation(rows)

    assert summary["accuracy"] == 1.0
    assert summary["by_category"]["answerable"] == {"correct": 1, "total": 1}
    assert evaluation_csv(rows).startswith(b"\xef\xbb\xbf")


def test_one_evaluation_error_does_not_stop_the_remaining_questions():
    rows = run_evaluation(
        FakeQAService(),
        [question("answerable", "error"), question("unrelated", "unrelated")],
        top_k=5,
        distance_threshold=0.45,
    )

    assert [row["final_status"] for row in rows] == ["error", "low_relevance"]
    assert rows[1]["correct"] is True
