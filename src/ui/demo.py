"""3〜5分のゼミ発表に使うデモ準備と代表シナリオ。"""

from __future__ import annotations

import time

import streamlit as st

from src.demo_data import register_demo_documents
from src.models import RegistrationStatus
from src.runtime import ApplicationServices
from src.ui.common import (
    queue_question_scenario,
    render_chips,
    render_page_header,
    show_application_error,
)

_SCENARIOS = (
    {
        "label": "回答できる",
        "question": "小学生にも分かるように、月の満ち欠けを説明してください",
        "visitor": "小学生",
        "scene": "展示前で短く説明",
        "language": "やさしい日本語",
        "tone": "safe",
    },
    {
        "label": "関連度で停止",
        "question": "徳川家康について説明してください",
        "visitor": "一般の大人",
        "scene": "質問に詳しく回答",
        "language": "日本語",
        "tone": "stop",
    },
    {
        "label": "根拠不足で停止",
        "question": "月の裏側には宇宙人がいるのですか",
        "visitor": "小学生",
        "scene": "展示前で短く説明",
        "language": "やさしい日本語",
        "tone": "stop",
    },
)


def _counts(services: ApplicationServices) -> tuple[int, int]:
    records = [
        record
        for record in services.repository.list_documents()
        if record.status == "registered"
    ]
    return len(records), sum(record.chunk_count for record in records)


def render(services: ApplicationServices) -> None:
    render_page_header("デモ")

    document_count, chunk_count = _counts(services)
    render_chips((f"資料 {document_count}件", f"検索区画 {chunk_count}件"))
    register_column, warmup_column = st.columns(2)
    register = register_column.button(
        "デモ資料を準備",
        type="primary",
        use_container_width=True,
    )
    warmup = warmup_column.button("回答モデルを準備", use_container_width=True)

    if register:
        with st.spinner("デモ資料を準備しています…"):
            summary = register_demo_documents(
                services.registration_service,
                services.config.project_root / "sample_docs",
            )
        registered = sum(
            result.status in {RegistrationStatus.REGISTERED, RegistrationStatus.REPLACED}
            for result in summary.results
        )
        duplicates = sum(
            result.status == RegistrationStatus.ALREADY_REGISTERED
            for result in summary.results
        )
        if registered:
            st.success(f"{registered}件を登録しました。")
        elif duplicates:
            st.info("デモ資料は登録済みです。")
        for error in summary.errors:
            st.error(f"登録できない資料がありました：{error}")

    if warmup:
        started = time.perf_counter()
        try:
            with st.spinner("回答モデルを準備しています…"):
                services.generation_client.warmup()
            st.success(f"準備完了（{time.perf_counter() - started:.1f}秒）")
        except Exception as exc:
            show_application_error(exc)

    st.markdown("### シナリオ")
    columns = st.columns(3)
    for index, (column, scenario) in enumerate(
        zip(columns, _SCENARIOS, strict=True), start=1
    ):
        with column.container(border=True):
            st.markdown(
                f'<div class="action-number">{index:02}</div>'
                f'<div class="result-tag {scenario["tone"]}">'
                f'<span class="status-dot"></span>{scenario["label"]}</div>'
                f'<div class="scenario-question">{scenario["question"]}</div>',
                unsafe_allow_html=True,
            )
            render_chips((scenario["visitor"], scenario["language"]))
            if st.button(
                "この質問を試す",
                key=f"set_demo_scenario_{index}",
                use_container_width=True,
            ):
                queue_question_scenario(
                    question=scenario["question"],
                    visitor=scenario["visitor"],
                    scene=scenario["scene"],
                    language=scenario["language"],
                )
                st.rerun()

    st.caption("デモ資料は架空です。")
