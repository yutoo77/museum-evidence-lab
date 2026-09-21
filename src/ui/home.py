"""ホーム画面。"""

from __future__ import annotations

import streamlit as st

from src.runtime import ApplicationServices
from src.ui.common import go_to, render_overview_stats, render_page_header


def render(services: ApplicationServices) -> None:
    render_page_header("ホーム")

    records = services.repository.list_documents()
    registered = sum(record.status == "registered" for record in records)
    total_chunks = sum(
        record.chunk_count for record in records if record.status == "registered"
    )
    render_overview_stats(
        [
            ("登録資料", f"{registered}件"),
            ("検索区画", f"{total_chunks}件"),
            ("処理", "PC内"),
        ]
    )

    ask_column, demo_column, register_column = st.columns([1.7, 1, 1])
    if ask_column.button(
        "質問する",
        type="primary",
        use_container_width=True,
    ):
        go_to("質問する")
    if demo_column.button("デモ", use_container_width=True):
        go_to("デモ")
    if register_column.button("資料を登録", use_container_width=True):
        go_to("資料を登録")
