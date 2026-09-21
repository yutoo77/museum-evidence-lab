from __future__ import annotations

import importlib.util

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("chromadb") is None,
    reason="Optional old-app UI: install requirements-legacy.txt; lab UI has separate tests",
)


def test_streamlit_home_starts_without_uncaught_exception():
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "ホーム"


@pytest.mark.parametrize(
    ("navigation", "title"),
    (
        ("デモ", "デモ"),
        ("資料を登録", "資料を登録"),
        ("資料一覧", "資料一覧"),
        ("質問する", "質問する"),
        ("簡易評価", "簡易評価"),
        ("利用履歴・フィードバック", "利用履歴"),
        ("設定・管理", "設定・管理"),
    ),
)
def test_streamlit_page_starts_without_uncaught_exception(
    navigation: str,
    title: str,
) -> None:
    app = AppTest.from_file("app.py", default_timeout=30)
    app.session_state["main_navigation"] = navigation
    app.run()

    assert not app.exception
    assert app.title[0].value == title
