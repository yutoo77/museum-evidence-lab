"""各画面で共有する表示処理。"""

from __future__ import annotations

import logging
from html import escape

import streamlit as st

from src.config import AppConfig
from src.exceptions import RAGApplicationError

logger = logging.getLogger(__name__)

VISITOR_PROFILES = ("小学生", "中高生", "一般の大人", "科学に詳しい人", "外国人来館者")
EXPLANATION_SCENES = (
    "展示前で短く説明",
    "展示を見ながら説明",
    "質問に詳しく回答",
    "FAQの下書き",
)
RESPONSE_LANGUAGES = ("日本語", "やさしい日本語", "英語", "中国語")

NAVIGATION_LABELS = {
    "ホーム": "⌂  ホーム",
    "デモ": "▶  デモ",
    "資料を登録": "＋  資料を登録",
    "資料一覧": "▤  資料一覧",
    "質問する": "⌕  質問する",
    "簡易評価": "✓  簡易評価",
    "利用履歴・フィードバック": "◷  利用履歴",
    "設定・管理": "⚙  設定・管理",
}


def apply_pending_state() -> None:
    pending = st.session_state.pop("pending_qa_settings", None)
    if isinstance(pending, dict):
        for key, value in pending.items():
            st.session_state[key] = value
    pending_page = st.session_state.pop("pending_navigation", None)
    if pending_page:
        st.session_state["main_navigation"] = pending_page


def initialize_session_state(config: AppConfig) -> None:
    defaults = {
        "qa_question": "",
        "qa_visitor_profile": "一般の大人",
        "qa_explanation_scene": "質問に詳しく回答",
        "qa_response_language": "日本語",
        "qa_top_k": config.retrieval.top_k,
        "qa_distance_threshold": config.retrieval.distance_threshold,
        "last_answer_result": None,
        "evaluation_results": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def queue_question_scenario(
    *, question: str, visitor: str, scene: str, language: str
) -> None:
    st.session_state["pending_qa_settings"] = {
        "qa_question": question,
        "qa_visitor_profile": visitor,
        "qa_explanation_scene": scene,
        "qa_response_language": language,
        "last_answer_result": None,
    }
    st.session_state["pending_navigation"] = "質問する"


def render_sidebar_settings(config: AppConfig) -> None:
    with st.sidebar.expander("回答条件"):
        st.selectbox(
            "来館者",
            VISITOR_PROFILES,
            key="qa_visitor_profile",
        )
        st.selectbox(
            "説明する場面",
            EXPLANATION_SCENES,
            key="qa_explanation_scene",
        )
        st.selectbox(
            "回答言語",
            RESPONSE_LANGUAGES,
            key="qa_response_language",
        )
    with st.sidebar.expander("検索設定"):
        st.slider(
            "候補数",
            min_value=1,
            max_value=10,
            key="qa_top_k",
        )
        st.number_input(
            "関連度の基準",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            key="qa_distance_threshold",
        )


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="brand-mark">M</div>
          <div>
            <div class="brand-name">Museum Evidence<br>Assistant</div>
            <div class="brand-subtitle">資料検索・回答支援</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str) -> None:
    st.title(title)


def render_overview_stats(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="overview-stat"><div class="overview-value">{escape(value)}</div>'
        f'<div class="overview-name">{escape(label)}</div></div>'
        for label, value in items
    )
    st.markdown(
        '<div class="overview-panel">'
        '<div class="overview-identity"><span class="overview-pulse"></span>'
        '<div><div class="overview-code">LOCAL RAG</div>'
        '<div class="overview-small">館内資料のみ</div></div></div>'
        f'<div class="overview-stats">{cells}</div></div>',
        unsafe_allow_html=True,
    )


def render_chips(items: list[str] | tuple[str, ...]) -> None:
    chips = "".join(f'<span class="ui-chip">{escape(item)}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def go_to(page: str) -> None:
    st.session_state["pending_navigation"] = page
    st.rerun()


def show_application_error(error: Exception) -> None:
    if isinstance(error, RAGApplicationError):
        user_message = error.user_message
        detail = error.technical_detail
    else:
        user_message = "予期しないエラーが発生しました。管理者に連絡してください。"
        detail = f"{type(error).__name__}: {error}"
    logger.error("UI operation failed: %s", detail)
    st.error(user_message)
    with st.expander("管理者向けの詳細"):
        st.code(detail, language="text")


def apply_accessible_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --md-blue: #2F6FEB;
            --md-blue-dark: #2459BD;
            --md-navy: #E6EEF8;
            --md-ink: #111C2E;
            --md-muted: #66758B;
            --md-line: #D8E1EC;
            --md-surface: #FFFFFF;
        }
        html, body, .stApp {
            font-family: Inter, "Noto Sans JP", "Yu Gothic UI", Meiryo, sans-serif;
        }
        .stApp {
            color: var(--md-ink);
            font-size: 16px;
            background: #F3F6FA;
        }
        [data-testid="stDecoration"], footer { display: none; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { background: transparent; }
        [data-testid="stToolbarActions"],
        [data-testid="stHeaderActionElements"],
        [data-testid="stAppDeployButton"],
        [data-testid="stMainMenu"] { display: none; }
        [data-testid="stExpandSidebarButton"] { display: flex !important; }
        .block-container {
            max-width: 1120px;
            padding-top: 2.6rem;
            padding-bottom: 4rem;
        }
        h1 {
            color: #0D182A;
            font-size: 2rem !important;
            letter-spacing: -0.035em;
            margin: 0 0 1.35rem !important;
        }
        h2, h3 { color: #17243A; letter-spacing: -0.02em; }
        p, li { line-height: 1.65; }
        [data-testid="stCaptionContainer"] { color: var(--md-muted); }
        [data-testid="stSidebar"] {
            background: var(--md-navy);
            border-right: 1px solid #D3DFED;
            box-shadow: 8px 0 26px rgba(45, 73, 107, 0.08);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        [data-testid="stSidebar"] hr { border-color: #CCD9E8; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.18rem; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 0.58rem 0.75rem;
            border-radius: 0.55rem;
            transition: background 140ms ease, color 140ms ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #D9E5F3;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child {
            display: none;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div {
            gap: 0 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #2F6FEB;
            box-shadow: 0 7px 17px rgba(47, 111, 235, 0.22);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label p,
        [data-testid="stSidebar"] summary p,
        [data-testid="stSidebar"] summary span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {
            color: #425976 !important;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
            color: #FFFFFF !important;
            font-weight: 720;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 0;
            background: transparent;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding: 0.2rem 0.35rem 1.2rem;
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 2.35rem;
            height: 2.35rem;
            color: white;
            background: linear-gradient(145deg, #5689F0, #2F6FEB);
            border-radius: 0.72rem;
            font-weight: 800;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
        }
        .brand-name {
            color: #1B304C;
            font-size: 0.9rem;
            font-weight: 750;
            line-height: 1.2;
        }
        .brand-subtitle { color: #73869F; font-size: 0.72rem; margin-top: 0.05rem; }
        div.stButton > button,
        div.stDownloadButton > button {
            min-height: 2.75rem;
            border-radius: 0.55rem;
            border-color: #CBD6E4;
            font-weight: 680;
            box-shadow: none;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            border-color: #2563EB;
            color: #1D4ED8;
        }
        button[kind="primary"] {
            background: #2F6FEB !important;
            border-color: #2F6FEB !important;
            color: white !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.18) !important;
        }
        [data-testid="stMetric"] { min-height: 6.4rem; padding: 0.9rem 1rem; background: #FFFFFF; border: 1px solid var(--md-line); border-radius: 0.65rem; }
        [data-testid="stMetricValue"] { color: #17243A; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border-color: var(--md-line);
            border-radius: 0.7rem;
            box-shadow: none;
        }
        [data-testid="stFileUploader"] {
            padding: 0.6rem;
            border-radius: 0.85rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--md-line);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        textarea, input { border-radius: 0.6rem !important; }
        .trust-item {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.7rem;
            color: #40516A;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--md-line);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .trust-item span { color: #3B82F6; font-size: 0.52rem; }
        [data-testid="stSidebar"] .trust-item {
            color: #4C6687;
            background: rgba(255, 255, 255, 0.48);
            border-color: #CAD8E8;
        }
        .overview-panel {
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 1.5rem;
            min-height: 8.5rem;
            margin: 0.2rem 0 1rem;
            padding: 1.5rem 1.65rem;
            color: #FFFFFF;
            background:
                radial-gradient(circle at 92% 0%, rgba(181, 211, 255, 0.34), transparent 18rem),
                linear-gradient(125deg, #315F9F, #4B7FC2);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.85rem;
            box-shadow: 0 16px 36px rgba(48, 91, 151, 0.18);
        }
        .overview-identity { display: flex; align-items: center; gap: 0.7rem; min-width: 9rem; }
        .overview-pulse { width: 0.62rem; height: 0.62rem; border-radius: 50%; background: #C9DFFF; box-shadow: 0 0 0 5px rgba(213, 230, 255, 0.2); }
        .overview-code { font-size: 0.75rem; font-weight: 820; letter-spacing: 0.14em; }
        .overview-small { margin-top: 0.18rem; color: #D4E2F5; font-size: 0.72rem; }
        .overview-stats { display: flex; align-items: stretch; }
        .overview-stat { min-width: 8.5rem; padding: 0.2rem 1.45rem; border-left: 1px solid rgba(255, 255, 255, 0.13); }
        .overview-value { font-size: 2rem; font-weight: 720; letter-spacing: -0.04em; line-height: 1.2; }
        .overview-name { margin-top: 0.35rem; color: #D8E4F4; font-size: 0.75rem; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.55rem 0 1rem; }
        .ui-chip {
            display: inline-flex;
            padding: 0.34rem 0.65rem;
            color: #31517D;
            background: #EDF4FF;
            border: 1px solid #D7E6FB;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .action-number { color: #2563EB; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; }
        .scenario-question { min-height: 5rem; color: #17243A; font-weight: 680; line-height: 1.55; }
        .status-card {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.8rem 0.9rem;
            margin: 0.65rem 0 1rem;
            border: 1px solid #D9E5F4;
            border-radius: 0.75rem;
            background: #F5F9FF;
            color: #28486F;
            font-weight: 680;
        }
        .status-card.safe { background: #F3F7FF; border-color: #D6E3F8; }
        .status-card.stop { background: #FFF9ED; border-color: #F2E2BC; color: #765B20; }
        .status-card.error { background: #FFF4F4; border-color: #F2D1D1; color: #8A3434; }
        .status-dot { width: 0.55rem; height: 0.55rem; border-radius: 50%; background: #3B82F6; flex: 0 0 auto; }
        .status-card.stop .status-dot { background: #D89B23; }
        .status-card.error .status-dot { background: #D85A5A; }
        .result-tag {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            margin: 0.7rem 0 1rem;
            padding: 0.34rem 0.58rem;
            color: #24509A;
            background: #EAF1FF;
            border-radius: 0.35rem;
            font-size: 0.76rem;
            font-weight: 720;
        }
        .result-tag.stop { color: #765B20; background: #FFF3D7; }
        .result-tag .status-dot { width: 0.42rem; height: 0.42rem; }
        @media (max-width: 700px) {
            .block-container { padding-top: 1.4rem; }
            h1 { font-size: 1.8rem !important; }
            .overview-panel { align-items: flex-start; flex-direction: column; }
            .overview-stats { width: 100%; }
            .overview-stat { min-width: 0; flex: 1; padding: 0.2rem 0.7rem; }
            .overview-stat:first-child { border-left: 0; padding-left: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
