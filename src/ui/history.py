"""後方互換な質問履歴とフィードバック表示。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.runtime import ApplicationServices
from src.ui.common import render_page_header

_STATUS_LABELS = {
    "answered": "回答成功",
    "low_relevance": "関連度不足",
    "unanswerable": "資料だけでは回答不能",
    "error": "エラー",
}


def _source_names(value: Any) -> str:
    if not isinstance(value, list):
        return "－"
    names: list[str] = []
    for source in value:
        if isinstance(source, dict):
            name = str(source.get("source_name", ""))
        else:
            name = str(source)
        if name and name not in names:
            names.append(name)
    return "、".join(names) or "－"


def _sources(item: dict[str, Any]) -> Any:
    return item.get("sources", item.get("used_sources", item.get("evidence", [])))


def _feedback(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("feedback")
    if isinstance(value, dict):
        return value
    rating = item.get("feedback_rating", item.get("rating"))
    comment = item.get("feedback_comment", item.get("comment", ""))
    return {"rating": rating, "comment": comment} if rating else {}


def _status(item: dict[str, Any]) -> str:
    return str(item.get("status", item.get("answer_status", "")))


def render(services: ApplicationServices) -> None:
    render_page_header("利用履歴")
    interactions, corrupted = services.interaction_log.list_interactions()
    if corrupted:
        st.warning(f"読み取れないログ行が{corrupted}件ありました。ほかの履歴は表示しています。")
    if not interactions:
        st.info("質問履歴はまだありません。")
        return

    rows = []
    for item in interactions:
        feedback = _feedback(item)
        status = _status(item)
        rows.append(
            {
                "日時": str(item.get("timestamp", "")).replace("T", " ").replace("Z", " UTC"),
                "質問": item.get("question", ""),
                "状態": _STATUS_LABELS.get(status, status or "－"),
                "資料": _source_names(_sources(item)),
                "評価": feedback.get("rating", "－") if isinstance(feedback, dict) else "－",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.markdown("#### 詳細")
    for item in interactions[:12]:
        question = str(item.get("question", "質問なし"))
        status = _status(item)
        label = _STATUS_LABELS.get(status, status or "－")
        with st.expander(f"{label}　{question[:64]}"):
            st.write(item.get("answer", "回答記録なし"))
            st.caption(f"使用資料：{_source_names(_sources(item))}")
            feedback = _feedback(item)
            if feedback:
                st.write(f"職員評価：{feedback.get('rating', '－')}")
                if feedback.get("comment"):
                    st.write(f"コメント：{feedback['comment']}")
