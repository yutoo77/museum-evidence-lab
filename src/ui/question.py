"""質問応答、根拠表示、職員フィードバック画面。"""

from __future__ import annotations

import streamlit as st

from src.models import AnswerResult, AnswerStatus, QuestionOptions, SearchHit
from src.runtime import ApplicationServices
from src.ui.common import (
    go_to,
    render_chips,
    render_page_header,
    show_application_error,
)

_FEEDBACK_OPTIONS = (
    "そのまま使える",
    "少し修正すれば使える",
    "根拠が弱い",
    "資料外の内容を含んでいる",
    "使えない",
)


def _options() -> QuestionOptions:
    return QuestionOptions(
        visitor_profile=st.session_state.qa_visitor_profile,
        explanation_scene=st.session_state.qa_explanation_scene,
        response_language=st.session_state.qa_response_language,
        top_k=int(st.session_state.qa_top_k),
        distance_threshold=float(st.session_state.qa_distance_threshold),
    )


def _source_label(hit: SearchHit) -> str:
    page = f"p.{hit.page_number}" if hit.page_number is not None else "ページなし"
    return f"{hit.source_name} / {page}"


def _render_status(result: AnswerResult) -> None:
    if result.status == AnswerStatus.ANSWERED:
        tone = "safe"
        message = "回答案を作成しました。根拠と照合してください。"
    elif result.status == AnswerStatus.LOW_RELEVANCE:
        tone = "stop"
        message = "関連する資料がないため、回答を停止しました。"
    elif result.status == AnswerStatus.UNANSWERABLE:
        tone = "stop"
        message = "資料だけでは確認できないため、回答を停止しました。"
    else:
        tone = "error"
        message = "処理できませんでした。設定・管理を確認してください。"
    st.markdown(
        f'<div class="status-card {tone}"><span class="status-dot"></span>{message}</div>',
        unsafe_allow_html=True,
    )


def _render_evidence(result: AnswerResult) -> None:
    st.markdown("#### 根拠資料")
    if not result.evidence:
        st.caption("根拠候補はありません。")
        return
    for index, hit in enumerate(result.evidence, start=1):
        with st.expander(f"{index}. {_source_label(hit)}"):
            st.write(hit.text)
            st.caption(
                f"区画 {hit.chunk_number}　関連距離 {hit.distance:.4f}"
            )


def _render_feedback(services: ApplicationServices, result: AnswerResult) -> None:
    rating_key = f"feedback_rating_{result.interaction_id}"
    comment_key = f"feedback_comment_{result.interaction_id}"
    saved_key = f"feedback_saved_{result.interaction_id}"
    with st.expander("この回答を評価"):
        rating = st.selectbox(
            "評価",
            _FEEDBACK_OPTIONS,
            index=None,
            placeholder="選択してください",
            key=rating_key,
        )
        comment = st.text_area(
            "コメント（任意）",
            key=comment_key,
            placeholder="気になった点を入力",
            height=90,
        )
        if st.button(
            "保存",
            disabled=rating is None,
            key=f"save_feedback_{result.interaction_id}",
        ):
            try:
                services.interaction_log.save_feedback(result, str(rating), comment)
                st.session_state[saved_key] = True
            except Exception as exc:
                show_application_error(exc)
        if st.session_state.get(saved_key):
            st.success("保存しました。")


def _render_result(services: ApplicationServices, result: AnswerResult) -> None:
    _render_status(result)
    answer_column, evidence_column = st.columns([1.2, 1])
    with answer_column.container(border=True):
        st.markdown("#### 回答案")
        st.markdown(result.answer)
        st.caption(f"{result.elapsed_seconds:.2f}秒")
    with evidence_column:
        _render_evidence(result)
    _render_feedback(services, result)
    with st.expander("技術情報の詳細"):
        st.write(f"最終ステータス：`{result.status.value}`")
        st.write(
            "最小distance："
            + (f"{result.min_distance:.4f}" if result.min_distance is not None else "結果なし")
        )
        st.write(f"資料との関連度の基準：{result.distance_threshold:.2f}")
        st.write(f"回答可能性判定：`{result.answerability}`")
        st.write(f"検索用モデル：`{result.embedding_model}`")
        st.write(f"回答生成モデル：`{result.generation_model}`")
        if result.error_detail:
            st.code(result.error_detail, language="text")


def render(services: ApplicationServices) -> None:
    render_page_header("質問する")
    records = [
        record for record in services.repository.list_documents() if record.status == "registered"
    ]
    if not records:
        st.markdown(
            '<div class="status-card stop"><span class="status-dot"></span>'
            "登録資料がありません。</div>",
            unsafe_allow_html=True,
        )
        if st.button("デモ資料を用意する", type="primary"):
            go_to("デモ")
        return

    render_chips(
        (
            st.session_state.qa_visitor_profile,
            st.session_state.qa_explanation_scene,
            st.session_state.qa_response_language,
        )
    )
    st.text_area(
        "質問",
        key="qa_question",
        height=120,
        placeholder="例：小学生にも分かるように、月の満ち欠けを説明してください",
    )
    ask = st.button(
        "資料を調べる",
        type="primary",
        disabled=not st.session_state.qa_question.strip(),
        use_container_width=True,
    )
    if ask:
        st.session_state.last_answer_result = None
        with st.status("資料を確認しています…") as status:
            result = services.qa_service.answer(
                st.session_state.qa_question,
                _options(),
                progress=lambda message: status.write(message),
            )
            if result.status == AnswerStatus.ANSWERED:
                status.update(label="回答案を作成しました", state="complete", expanded=False)
            elif result.status == AnswerStatus.ERROR:
                status.update(label="処理できませんでした", state="error", expanded=False)
            else:
                status.update(label="回答を停止しました", state="complete", expanded=False)
        st.session_state.last_answer_result = result

    result = st.session_state.get("last_answer_result")
    if isinstance(result, AnswerResult):
        _render_result(services, result)
