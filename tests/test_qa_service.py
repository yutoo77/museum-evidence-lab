from __future__ import annotations

from collections import deque

from src.exceptions import OllamaConnectionError
from src.interaction_logs import InteractionLogStore
from src.models import AnswerStatus, QuestionOptions, SearchHit
from src.qa_service import QuestionAnsweringService, parse_answerability
from src.retrieval import RetrievalOutcome


class FakeRetrieval:
    def __init__(self, outcome=None, error=None):
        self.outcome = outcome
        self.error = error

    def search(self, question, top_k, threshold):
        if self.error:
            raise self.error
        return self.outcome


class FakeGeneration:
    model_name = "fake-qwen"

    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def chat(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt, kwargs))
        return self.responses.popleft()


def options():
    return QuestionOptions("小学生", "展示前で短く説明", "やさしい日本語", 5, 0.45)


def evidence(distance=0.2):
    return (
        SearchHit(
            "月は太陽の光を反射して見えます。",
            distance,
            "moon_phase.txt",
            None,
            1,
            "doc",
            "version",
        ),
    )


def service(tmp_path, outcome, responses=(), error=None):
    generation = FakeGeneration(responses)
    qa = QuestionAnsweringService(
        FakeRetrieval(outcome, error),
        generation,
        InteractionLogStore(tmp_path / "interactions.jsonl"),
        "fake-embedding",
    )
    return qa, generation


def test_low_relevance_stops_before_llm(tmp_path):
    qa, generation = service(tmp_path, RetrievalOutcome(evidence(0.8), 0.8, False))

    result = qa.answer("徳川家康とは", options())

    assert result.status == AnswerStatus.LOW_RELEVANCE
    assert result.answerability == "NOT_RUN"
    assert generation.calls == []


def test_answerability_no_stops_answer_generation(tmp_path):
    qa, generation = service(tmp_path, RetrievalOutcome(evidence(), 0.2, True), ["NO"])

    result = qa.answer("月の裏側に宇宙人はいますか", options())

    assert result.status == AnswerStatus.UNANSWERABLE
    assert result.answerability == "NO"
    assert len(generation.calls) == 1


def test_invalid_answerability_is_conservatively_stopped(tmp_path):
    qa, _ = service(tmp_path, RetrievalOutcome(evidence(), 0.2, True), ["たぶんYESです"])

    result = qa.answer("月について", options())

    assert result.status == AnswerStatus.UNANSWERABLE
    assert result.answerability == "INVALID"


def test_answer_uses_profile_scene_language_and_evidence(tmp_path):
    qa, generation = service(
        tmp_path,
        RetrievalOutcome(evidence(), 0.2, True),
        ["YES", "月は、太陽の光をうつして光って見えます。"],
    )

    result = qa.answer("月が光る理由は", options())

    assert result.status == AnswerStatus.ANSWERED
    assert result.answerability == "YES"
    assert "太陽の光" in result.answer
    answer_prompt = generation.calls[1][1]
    assert "小学4〜6年生" in answer_prompt
    assert "3〜5文" in answer_prompt
    assert "やさしい日本語" in answer_prompt
    assert "moon_phase.txt" in answer_prompt


def test_ollama_error_becomes_error_result_and_does_not_escape(tmp_path):
    qa, _ = service(
        tmp_path,
        None,
        error=OllamaConnectionError("Ollamaが停止しています。", "connection refused"),
    )

    result = qa.answer("質問", options())

    assert result.status == AnswerStatus.ERROR
    assert result.answer == "Ollamaが停止しています。"
    assert result.error_detail == "connection refused"


def test_answerability_parser_accepts_only_exact_yes_or_no():
    assert parse_answerability(" YES。\n") == "YES"
    assert parse_answerability("NO") == "NO"
    assert parse_answerability("YES because") == "INVALID"
