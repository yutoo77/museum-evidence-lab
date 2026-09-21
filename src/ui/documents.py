"""登録済み資料一覧画面。"""

from __future__ import annotations

import streamlit as st

from src.models import DocumentRecord
from src.runtime import ApplicationServices
from src.ui.common import render_page_header

_STATUS_LABELS = {
    "registered": "登録済み",
    "processing": "処理中",
    "failed": "登録失敗",
}


def _page_count(record: DocumentRecord) -> str | int:
    return record.page_count if record.file_type == "pdf" else "－"


def render(services: ApplicationServices) -> None:
    render_page_header("資料一覧")
    records = services.repository.list_documents()
    if not records:
        st.info("登録済みの資料はありません。「資料を登録」から追加してください。")
        return

    rows = [
        {
            "資料名": record.source_name,
            "形式": record.file_type.upper(),
            "ページ数": _page_count(record),
            "文章のまとまり": record.chunk_count,
            "登録日時": record.registered_at.replace("T", " ").replace("Z", " UTC"),
            "状態": _STATUS_LABELS.get(record.status, record.status),
        }
        for record in records
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    warnings = [record for record in records if record.warning]
    if warnings:
        with st.expander("登録時の注意事項"):
            for record in warnings:
                st.warning(f"{record.source_name}：{record.warning}")
