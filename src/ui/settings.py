"""ローカルモデルと主要設定の確認画面。"""

from __future__ import annotations

from html import escape

import streamlit as st

from src.runtime import ApplicationServices
from src.ui.common import render_page_header


def render(services: ApplicationServices) -> None:
    render_page_header("設定・管理")
    try:
        models = services.embedding_client.list_models()
        embedding_ok = any(
            model.split(":", 1)[0] == services.config.ollama.embedding_model.split(":", 1)[0]
            for model in models
        )
        generation_ok = services.config.ollama.generation_model in models
        left, right = st.columns(2)
        statuses = (
            (left, "検索モデル", services.config.ollama.embedding_model, embedding_ok),
            (right, "回答モデル", services.config.ollama.generation_model, generation_ok),
        )
        for column, label, model, available in statuses:
            tone = "safe" if available else "error"
            suffix = "利用可能" if available else "未導入"
            column.markdown(
                f'<div class="status-card {tone}"><span class="status-dot"></span>'
                f'{label}　<strong>{escape(model)}</strong>　{suffix}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.error("Ollamaに接続できません。起動状態を確認してください。")

    with st.expander("モデルを追加する"):
        st.code(
            "ollama list\nollama pull embeddinggemma\nollama pull qwen3:1.7b",
            language="powershell",
        )
    with st.expander("現在の主要設定"):
        st.write(f"Ollama接続先：`{services.config.ollama.base_url}`")
        st.write(f"文章の分割：{services.config.chunking.chunk_size}文字")
        st.write(f"重なり：{services.config.chunking.chunk_overlap}文字")
        st.write(f"既定の候補数：{services.config.retrieval.top_k}")
        st.write(f"既定の関連度基準：{services.config.retrieval.distance_threshold:.2f}")
        st.caption("質問ごとの候補数と関連度基準はサイドバーで調整できます。")

    with st.expander("このアプリについて"):
        st.markdown(
            "Museum Evidence Assistant  \n"
            "Copyright © 2026 yutoo77  \n"
            "GNU Affero General Public License v3.0で公開されています。  \n"
            "本ソフトウェアは無保証の研究プロトタイプです。"
        )
        st.markdown(
            "[Source code・ライセンスを確認する]"
            "(https://github.com/yutoo77/museum-evidence-assistant)"
        )
