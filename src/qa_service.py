"""検索、資料外判定、回答可能性判定、回答生成を一つの流れで実行する。"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable

from src.exceptions import RAGApplicationError
from src.generation import GenerationProvider
from src.interaction_logs import InteractionLogStore, utc_now
from src.models import AnswerResult, AnswerStatus, QuestionOptions, SearchHit
from src.retrieval import RetrievalService

ProgressCallback = Callable[[str], None]


_PROFILE_GUIDANCE = {
    "小学生": "小学4〜6年生が分かる短い文で、難しい語には言い換えを添える。",
    "中高生": "中高生向けに、原因と結果の関係が分かるように説明する。",
    "一般の大人": "専門知識を前提にせず、正確で簡潔に説明する。",
    "科学に詳しい人": "科学用語を適切に使い、資料にある範囲で詳しく説明する。",
    "外国人来館者": "文化固有の表現を避け、短く明確な文で説明する。",
}

_SCENE_GUIDANCE = {
    "展示前で短く説明": "展示を見る前の口頭説明として、3〜5文程度にまとめる。",
    "展示を見ながら説明": "展示物を指し示しながら話せる説明にする。",
    "質問に詳しく回答": "要点を先に示し、その後に根拠を補足する。",
    "FAQの下書き": "職員が修正しやすいFAQ回答案としてまとめる。",
}

_LANGUAGE_GUIDANCE = {
    "日本語": "自然で明確な日本語で回答する。",
    "やさしい日本語": "一文を短くし、難しい漢字や専門語を言い換えたやさしい日本語で回答する。",
    "英語": "Answer in clear English.",
    "中国語": "请用简明、准确的简体中文回答。",
}


def _context_text(hits: tuple[SearchHit, ...]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        page = f"p.{hit.page_number}" if hit.page_number is not None else "ページなし"
        blocks.append(
            f"[資料{index}: {hit.source_name} / {page} / チャンク{hit.chunk_number}]\n{hit.text}"
        )
    return "\n\n".join(blocks)


def parse_answerability(value: str) -> str:
    normalized = re.sub(r"[\s。.!！]", "", value).upper()
    if normalized == "YES":
        return "YES"
    if normalized == "NO":
        return "NO"
    return "INVALID"


class QuestionAnsweringService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_provider: GenerationProvider,
        log_store: InteractionLogStore,
        embedding_model: str,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_provider = generation_provider
        self.log_store = log_store
        self.embedding_model = embedding_model

    def answer(
        self,
        question: str,
        options: QuestionOptions,
        progress: ProgressCallback | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        interaction_id = str(uuid.uuid4())
        timestamp = utc_now()
        evidence: tuple[SearchHit, ...] = ()
        min_distance: float | None = None
        answerability = "NOT_RUN"

        def notify(message: str) -> None:
            if progress:
                progress(message)

        try:
            notify("1. 関連資料を検索中")
            outcome = self.retrieval_service.search(
                question, options.top_k, options.distance_threshold
            )
            evidence = outcome.hits
            min_distance = outcome.min_distance

            notify("2. 資料との関連度を確認中")
            if not outcome.is_relevant:
                result = self._result(
                    interaction_id,
                    timestamp,
                    question,
                    "登録資料内では確認できません。",
                    AnswerStatus.LOW_RELEVANCE,
                    evidence,
                    min_distance,
                    options,
                    answerability,
                    started,
                )
                self.log_store.save_interaction(result)
                return result

            notify("3. 資料だけで回答できるか確認中")
            context = _context_text(evidence)
            gate_response = self.generation_provider.chat(
                """あなたは資料根拠の厳格な判定器です。質問への答えが提示資料に明示されている場合だけYESと判定します。話題が近いだけ、一般知識や推測が必要、存在しないことを資料から断定できない場合はNOです。出力はYESまたはNOのどちらか1語だけにしてください。""",
                f"質問:\n{question.strip()}\n\n提示資料:\n{context}",
                temperature=0.0,
                max_tokens=8,
            )
            answerability = parse_answerability(gate_response)
            if answerability != "YES":
                answer = "登録資料だけでは回答できないため、回答を控えます。"
                if answerability == "INVALID":
                    answer = "回答可能性を安全に判定できなかったため、回答を控えます。"
                result = self._result(
                    interaction_id,
                    timestamp,
                    question,
                    answer,
                    AnswerStatus.UNANSWERABLE,
                    evidence,
                    min_distance,
                    options,
                    answerability,
                    started,
                )
                self.log_store.save_interaction(result)
                return result

            notify("4. 回答案を生成中")
            profile = _PROFILE_GUIDANCE.get(
                options.visitor_profile, _PROFILE_GUIDANCE["一般の大人"]
            )
            scene = _SCENE_GUIDANCE.get(
                options.explanation_scene, _SCENE_GUIDANCE["質問に詳しく回答"]
            )
            language = _LANGUAGE_GUIDANCE.get(
                options.response_language, _LANGUAGE_GUIDANCE["日本語"]
            )
            answer = self.generation_provider.chat(
                """あなたは科学館職員の回答案作成を支援します。提示資料だけを根拠にし、資料にない内容を補いません。科学的に不正確な断定を避け、思考過程は出力しません。回答は職員が確認・修正して使う下書きです。""",
                f"質問:\n{question.strip()}\n\n来館者への調整:\n{profile}\n{scene}\n{language}\n\n根拠資料:\n{context}\n\n上の条件に従って回答案だけを書いてください。",
                temperature=0.1,
                max_tokens=500,
            )
            result = self._result(
                interaction_id,
                timestamp,
                question,
                answer,
                AnswerStatus.ANSWERED,
                evidence,
                min_distance,
                options,
                answerability,
                started,
            )
        except Exception as exc:
            if isinstance(exc, RAGApplicationError):
                answer = exc.user_message
                detail = exc.technical_detail
            else:
                answer = "質問の処理中に予期しないエラーが発生しました。"
                detail = f"{type(exc).__name__}: {exc}"
            result = self._result(
                interaction_id,
                timestamp,
                question,
                answer,
                AnswerStatus.ERROR,
                evidence,
                min_distance,
                options,
                answerability,
                started,
                detail,
            )
        try:
            self.log_store.save_interaction(result)
        except OSError:
            pass
        return result

    def _result(
        self,
        interaction_id: str,
        timestamp: str,
        question: str,
        answer: str,
        status: AnswerStatus,
        evidence: tuple[SearchHit, ...],
        min_distance: float | None,
        options: QuestionOptions,
        answerability: str,
        started: float,
        error_detail: str = "",
    ) -> AnswerResult:
        return AnswerResult(
            interaction_id=interaction_id,
            timestamp=timestamp,
            question=question.strip(),
            answer=answer,
            status=status,
            evidence=evidence,
            min_distance=min_distance,
            distance_threshold=options.distance_threshold,
            answerability=answerability,
            visitor_profile=options.visitor_profile,
            explanation_scene=options.explanation_scene,
            response_language=options.response_language,
            embedding_model=self.embedding_model,
            generation_model=self.generation_provider.model_name,
            elapsed_seconds=round(time.perf_counter() - started, 3),
            error_detail=error_detail,
        )
