from __future__ import annotations

from src.interaction_logs import InteractionLogStore
from src.models import AnswerResult, AnswerStatus, SearchHit


def answer_result():
    return AnswerResult(
        interaction_id="interaction-1",
        timestamp="2026-08-04T00:00:00Z",
        question="月はなぜ光りますか",
        answer="太陽の光を反射するためです。",
        status=AnswerStatus.ANSWERED,
        evidence=(SearchHit("本文", 0.2, "moon.txt", None, 1, "doc", "v1"),),
        min_distance=0.2,
        distance_threshold=0.45,
        answerability="YES",
        visitor_profile="小学生",
        explanation_scene="展示前で短く説明",
        response_language="やさしい日本語",
        embedding_model="embeddinggemma",
        generation_model="qwen3:1.7b",
        elapsed_seconds=1.2,
    )


def test_interaction_and_feedback_are_merged(tmp_path):
    store = InteractionLogStore(tmp_path / "interactions.jsonl")
    result = answer_result()

    store.save_interaction(result)
    store.save_feedback(result, "そのまま使える", "分かりやすい")
    interactions, corrupted = store.list_interactions()

    assert corrupted == 0
    assert len(interactions) == 1
    assert interactions[0]["feedback"]["rating"] == "そのまま使える"
    assert interactions[0]["feedback"]["comment"] == "分かりやすい"


def test_old_and_corrupted_jsonl_lines_do_not_break_history(tmp_path):
    path = tmp_path / "interactions.jsonl"
    path.write_text(
        '{"interaction_id":"old-1","timestamp":"2025-01-01","question":"古い質問","answer":"古い回答"}\n'
        '{broken json\n',
        encoding="utf-8",
    )
    store = InteractionLogStore(path)

    interactions, corrupted = store.list_interactions()

    assert corrupted == 1
    assert interactions[0]["question"] == "古い質問"
    assert interactions[0].get("feedback") is None
