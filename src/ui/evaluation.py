"""9問の簡易評価とCSV出力画面。"""

from __future__ import annotations

import streamlit as st

from src.evaluation import (
    evaluation_csv,
    load_evaluation_questions,
    run_evaluation,
    summarize_evaluation,
)
from src.runtime import ApplicationServices
from src.ui.common import render_chips, render_page_header, show_application_error

_CATEGORY_LABELS = {
    "answerable": "資料だけで回答可能",
    "unrelated": "資料と無関係",
    "insufficient": "関連するが根拠不足",
}


def render(services: ApplicationServices) -> None:
    render_page_header("簡易評価")
    try:
        questions = load_evaluation_questions(
            services.config.project_root / "evaluation_questions.json"
        )
    except Exception as exc:
        show_application_error(exc)
        return

    counts = {category: 0 for category in _CATEGORY_LABELS}
    for item in questions:
        counts[str(item["category"])] += 1
    render_chips(
        tuple(
            f"{_CATEGORY_LABELS[category]} {count}問"
            for category, count in counts.items()
        )
    )
    if not services.repository.active_version_ids():
        st.info("評価前に「デモ」からデモ用資料を登録してください。")
        return

    if st.button("9問を評価", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0, text="評価を開始します…")
        status_text = st.empty()

        def update_progress(index: int, total: int, question: str) -> None:
            progress_bar.progress(
                (index - 1) / total,
                text=f"{index}/{total}問目を処理中",
            )
            status_text.caption(question)

        rows = run_evaluation(
            services.qa_service,
            questions,
            top_k=int(st.session_state.qa_top_k),
            distance_threshold=float(st.session_state.qa_distance_threshold),
            progress=update_progress,
        )
        progress_bar.progress(1.0, text="評価が完了しました。")
        status_text.empty()
        st.session_state.evaluation_results = rows

    rows = st.session_state.get("evaluation_results")
    if not isinstance(rows, list) or not rows:
        with st.expander("評価質問を確認"):
            for item in questions:
                st.write(f"- [{item['category']}] {item['question']}")
        return

    summary = summarize_evaluation(rows)
    first, second, third = st.columns(3)
    first.metric("全体正解率", f"{summary['accuracy'] * 100:.1f}%")
    second.metric("正解数", f"{summary['correct']} / {summary['total']}")
    third.metric("平均処理時間", f"{summary['average_seconds']:.2f}秒")

    render_chips(
        tuple(
            f"{_CATEGORY_LABELS.get(category, category)}：{values['correct']}/{values['total']}"
            for category, values in summary["by_category"].items()
        )
    )

    display_rows = [
        {
            "質問": row["question"],
            "期待分類": _CATEGORY_LABELS.get(row["expected_category"], row["expected_category"]),
            "最小distance": (
                round(row["min_distance"], 4) if row["min_distance"] is not None else "－"
            ),
            "distance判定": row["distance_decision"],
            "回答可能性": row["answerability"],
            "最終ステータス": row["final_status"],
            "一致": "○" if row["correct"] else "×",
            "処理時間（秒）": row["elapsed_seconds"],
        }
        for row in rows
    ]
    st.dataframe(display_rows, hide_index=True, use_container_width=True)
    st.download_button(
        "評価結果をCSVでダウンロード",
        data=evaluation_csv(rows),
        file_name="rag_evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if summary["errors"]:
        st.markdown("#### 要確認")
        for row in summary["errors"]:
            st.error(
                f"{row['question']}（期待：{row['expected_status']} ／ 実際：{row['final_status']}）"
            )
    else:
        st.markdown(
            '<div class="status-card safe"><span class="status-dot"></span>'
            "すべて期待どおりに判定されました。</div>",
            unsafe_allow_html=True,
        )
