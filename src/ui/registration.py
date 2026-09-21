"""資料登録画面。"""

from __future__ import annotations

import streamlit as st

from src.models import RegistrationStatus
from src.runtime import ApplicationServices
from src.ui.common import render_page_header, show_application_error


def render(services: ApplicationServices) -> None:
    render_page_header("資料を登録")

    uploaded_files = st.file_uploader(
        f"PDF・TXT・MD（1件 {services.config.app.max_upload_mb}MBまで）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="複数選択できます。画像だけのPDFには対応していません。",
    )

    if uploaded_files:
        total_mb = sum(uploaded.size for uploaded in uploaded_files) / (1024 * 1024)
        st.caption(f"{len(uploaded_files)}件を選択　合計 {total_mb:.1f}MB")

    register = st.button(
        "登録する",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )
    if not register:
        return

    progress = st.progress(0, text="資料の登録を始めます…")
    for index, uploaded in enumerate(uploaded_files):
        progress.progress(
            index / len(uploaded_files),
            text=f"「{uploaded.name}」を処理しています…",
        )
        try:
            result = services.registration_service.register_bytes(
                uploaded.name,
                uploaded.getvalue(),
            )
            if result.status == RegistrationStatus.ALREADY_REGISTERED:
                st.info(result.message)
            else:
                st.success(result.message)
                st.caption(
                    f"ページ数：{result.record.page_count}　／　"
                    f"検索できる文章：{result.record.chunk_count}件"
                )
                if result.record.warning:
                    st.warning(result.record.warning)
        except Exception as exc:
            show_application_error(exc)
    progress.progress(1.0, text="処理が完了しました。")
