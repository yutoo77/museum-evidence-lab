"""科学館職員向けローカルRAGのStreamlitエントリーポイント。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.runtime import ApplicationServices, build_services
from src.ui import (
    demo,
    documents,
    evaluation,
    history,
    home,
    question,
    registration,
    settings,
)
from src.ui.common import (
    NAVIGATION_LABELS,
    apply_accessible_styles,
    apply_pending_state,
    initialize_session_state,
    render_sidebar_brand,
    render_sidebar_settings,
    show_application_error,
)

st.set_page_config(
    page_title="科学館職員向け 資料検索・回答支援",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_accessible_styles()


@st.cache_resource
def services(config_revision: int, runtime_revision: int) -> ApplicationServices:
    """設定変更時だけ依存サービスを安全に組み立て直す。"""

    del config_revision, runtime_revision
    return build_services("config.yaml")


try:
    app_services = services(
        Path("config.yaml").stat().st_mtime_ns,
        Path("src/generation.py").stat().st_mtime_ns,
    )
except Exception as exc:
    st.title("科学館職員向け 資料検索・回答支援")
    show_application_error(exc)
    st.stop()

apply_pending_state()
initialize_session_state(app_services.config)

with st.sidebar:
    render_sidebar_brand()
selected_page = st.sidebar.radio(
    "利用する機能を選んでください",
    (
        "ホーム",
        "デモ",
        "資料を登録",
        "資料一覧",
        "質問する",
        "簡易評価",
        "利用履歴・フィードバック",
        "設定・管理",
    ),
    key="main_navigation",
    label_visibility="collapsed",
    format_func=lambda page: NAVIGATION_LABELS[page],
)
st.sidebar.divider()
render_sidebar_settings(app_services.config)
st.sidebar.divider()
st.sidebar.markdown(
    '<div class="trust-item"><span>●</span> このPC内で処理</div>',
    unsafe_allow_html=True,
)

if selected_page == "ホーム":
    home.render(app_services)
elif selected_page == "デモ":
    demo.render(app_services)
elif selected_page == "資料を登録":
    registration.render(app_services)
elif selected_page == "資料一覧":
    documents.render(app_services)
elif selected_page == "質問する":
    question.render(app_services)
elif selected_page == "簡易評価":
    evaluation.render(app_services)
elif selected_page == "利用履歴・フィードバック":
    history.render(app_services)
else:
    settings.render(app_services)
