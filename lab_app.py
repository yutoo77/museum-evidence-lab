"""Offline staff interface for the isolated evidence comparison application."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

from src.lab.contracts import LabAnswer
from src.lab.factory import build_lab
from src.lab.settings import load_lab_settings

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Museum Evidence Lab", page_icon="📘", layout="wide")
st.markdown(
    """<style>
.stApp{background:#fff;color:#1c2a3a}
[data-testid="stSidebar"]{background:#f7f9fc;border-right:1px solid #e3e9f1}
.block-container{max-width:1180px;padding-top:2.2rem;padding-bottom:3rem}
h1{font-size:1.8rem!important;letter-spacing:.01em}
[data-testid="stText"]{white-space:pre-wrap;line-height:1.8;font-family:inherit}
.stButton button[kind="primary"]{background:#2868c7;border-color:#2868c7}
[data-testid="stAppDeployButton"]{display:none}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def services(model, config_contents):
    return build_lab(ROOT / "lab_config.yaml", generation_model=model)


def bundle():
    return services(
        st.session_state.model, (ROOT / "lab_config.yaml").read_text(encoding="utf-8")
    )


def sources(evidence):
    if not evidence:
        st.caption("表示できる原文はありません。")
    for number, passage in enumerate(evidence, 1):
        page = (
            f" · {passage.page_number}ページ" if passage.page_number is not None else ""
        )
        # User-controlled names and contents are always plain text, never Markdown.
        with st.container(border=True):
            st.text(f"{number}. {passage.source_name}{page}")
            st.text(passage.text)


def local_time(value):
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone()
            .strftime("%Y/%m/%d %H:%M")
        )
    except (AttributeError, TypeError, ValueError):
        return str(value)


def render_answer(result):
    if result.status == "answered":
        label = "承認済みの回答" if result.route == "approved_faq" else "資料にある説明"
        if result.route == "explain":
            label = "説明案"
        elif result.route == "concise":
            label = "言い換え案 · 原文と照合してください"
        st.subheader(label)
        if result.route == "explain":
            st.caption("利用前に根拠をご確認ください。")
        reviewed_context = getattr(result, "reviewed_qa_context", ())
        if reviewed_context:
            st.caption("職員確認済みQ&Aを回答作成時の参考候補に含めました。")
            with st.expander("参考候補になった確認済みQ&A"):
                for item in reviewed_context:
                    st.text(f"質問: {item.question}")
                    st.text(f"確認済み回答: {item.approved_answer}")
                    st.text(f"確認日時: {local_time(item.created_at)}")
        for number, claim in enumerate(result.claims, 1):
            st.text(claim.text)
            if result.route in {"explain", "concise"}:
                with st.expander(f"{number}. この説明の根拠"):
                    cited_passages = {}
                    for reference in claim.references:
                        cited = next(
                            (
                                e
                                for e in result.evidence
                                if e.evidence_id == reference.evidence_id
                            ),
                            None,
                        )
                        if cited:
                            cited_passages.setdefault(cited.evidence_id, cited)
                            page = (
                                f" / {cited.page_number}ページ"
                                if cited.page_number is not None
                                else ""
                            )
                            st.text(f"出典: {cited.source_name}{page}")
                        else:
                            st.caption("出典を表示できません。")
                        st.text(reference.quote)
                    for context_number, cited in enumerate(cited_passages.values(), 1):
                        with st.expander(f"周辺の原文 {context_number}"):
                            page = (
                                f" / {cited.page_number}ページ"
                                if cited.page_number is not None
                                else ""
                            )
                            st.text(f"出典: {cited.source_name}{page}")
                            st.text(cited.text)
            else:
                for reference in claim.references:
                    cited = next(
                        (
                            e
                            for e in result.evidence
                            if e.evidence_id == reference.evidence_id
                        ),
                        None,
                    )
                    if cited:
                        page = (
                            f" / {cited.page_number}ページ"
                            if cited.page_number is not None
                            else ""
                        )
                        st.text(f"出典: {cited.source_name}{page}")
    else:
        st.text(result.message)
    st.caption(f"{result.timings.get('total_seconds', 0):.2f}秒")
    with st.expander("処理の内訳"):
        st.json(
            {
                "status": result.status,
                "route": result.route,
                "timings": result.timings,
                "calls": result.calls,
                "checks": result.issues,
            }
        )


def forget_answer():
    for key in (
        "last_result",
        "last_revision",
        "review_confirmed",
        "review_editor_open",
        "review_draft",
        "review_saved_id",
        "review_save_error",
    ):
        st.session_state.pop(key, None)


def render_review_actions(result):
    if st.session_state.get("review_saved_id") is not None:
        st.success("職員確認済みQ&Aとして保存しました。")
        return
    if st.session_state.get("review_save_error"):
        st.error(st.session_state.review_save_error)
    st.caption("回答と根拠を確認してから保存してください。")
    checked = st.checkbox("回答と根拠を確認しました", key="review_confirmed")
    approve_column, edit_column = st.columns(2)
    with approve_column:
        st.button(
            "この回答で問題ない",
            disabled=not checked or not st.session_state.get("last_revision"),
            width="stretch",
            on_click=save_reviewed_answer,
            args=(result, result.text),
        )
    with edit_column:
        if st.button("修正して保存", width="stretch"):
            st.session_state.review_editor_open = True
    if st.session_state.get("review_editor_open"):
        st.caption("修正後の内容も原文に照らしてください。")
        st.text_area(
            "修正後の回答",
            value=result.text,
            height=180,
            max_chars=803,
            key="review_draft",
        )
        st.button(
            "確認済みQ&Aとして保存",
            disabled=(
                not checked
                or not st.session_state.get("review_draft", "").strip()
                or not st.session_state.get("last_revision")
            ),
            on_click=save_reviewed_answer,
            args=(result,),
        )


def save_reviewed_answer(result, approved_answer=None):
    if approved_answer is None:
        approved_answer = st.session_state.get("review_draft", "")
    try:
        record = bundle().index.save_reviewed_qa(
            st.session_state.last_question,
            result,
            approved_answer,
            st.session_state.last_revision,
            audience=st.session_state.last_audience,
            detail=st.session_state.last_detail,
        )
    except ValueError:
        st.session_state.review_save_error = "保存できませんでした。回答と元資料を確認し、必要ならもう一度調べてください。"
    except Exception:
        st.session_state.review_save_error = (
            "保存に失敗しました。ローカルAIの接続を確認して、もう一度お試しください。"
        )
    else:
        st.session_state.pop("review_save_error", None)
        st.session_state.review_saved_id = record.id


settings = load_lab_settings(ROOT / "lab_config.yaml")
st.session_state.setdefault("model", settings.ollama.generation_model)
st.session_state.setdefault("ready", False)
with st.sidebar:
    st.markdown("**Museum Evidence Lab**")
    st.caption("科学館職員向け · ローカル比較版")
    page = st.radio(
        "画面",
        ["資料を調べる", "確認済みQ&A", "資料を管理", "比較結果", "設定"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("このPC内の資料を使用")
    if st.session_state.ready:
        st.caption("● 準備済み")
    else:
        st.caption("○ AIの準備が必要です")
    if st.button("AIを準備", icon=":material/power_settings_new:", width="stretch"):
        try:
            with st.status("ローカルAIを準備しています…") as status:
                current = bundle()
                current.embedding.warmup()
                current.client.warmup()
                st.session_state.ready = True
                status.update(label="準備できました", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.ready = False
            st.error("準備できませんでした。start_lab.cmdから起動してください。")
            st.caption(type(exc).__name__)

if page == "資料を調べる":
    st.title("資料を調べる")
    question = st.text_area(
        "質問",
        placeholder="例：月の満ち欠けは、なぜ起こる？",
        height=105,
        max_chars=1500,
    )
    labels = {
        "わかりやすく説明": "explain",
        "原文から回答": "quoted",
        "比較用（試験）": "comparison",
    }
    mode_column, audience_column, detail_column = st.columns([2, 1, 1])
    with mode_column:
        mode_label = st.selectbox("回答方法", list(labels), key="answer_mode")
    mode = labels[mode_label]
    if mode == "comparison":
        with st.expander("比較用の回答方法", expanded=True):
            comparison_labels = {
                "短く言い換え（試験）": "concise",
                "聞き返し・再検索（試験）": "bounded",
                "従来の2段階生成（比較用）": "baseline",
            }
            mode = comparison_labels[st.selectbox("比較方式", list(comparison_labels))]
    with audience_column:
        audience_labels = {
            "一般向け": "general",
            "子ども向け": "child",
            "職員向け": "staff",
        }
        audience = audience_labels[
            st.selectbox(
                "対象",
                list(audience_labels),
                disabled=mode != "explain",
                key="audience",
            )
        ]
    with detail_column:
        detail_labels = {"標準": "standard", "短く": "short"}
        detail = detail_labels[
            st.selectbox(
                "長さ", list(detail_labels), disabled=mode != "explain", key="detail"
            )
        ]
    if st.session_state.get("last_result") and (
        question != st.session_state.get("last_question")
        or mode != st.session_state.get("last_mode")
        or (
            mode == "explain"
            and (
                audience != st.session_state.get("last_audience")
                or detail != st.session_state.get("last_detail")
            )
        )
    ):
        forget_answer()
    ask = st.button(
        "調べる",
        type="primary",
        icon=":material/search:",
        disabled=not question.strip() or not st.session_state.ready,
    )
    if mode == "explain":
        answer_column = st.container()
        with st.expander("検索した原文"):
            evidence_box = st.empty()
    else:
        answer_column, source_column = st.columns([1, 1], gap="large")
        with source_column:
            st.subheader("根拠の原文")
            evidence_box = st.empty()
    if ask:
        forget_answer()
        current = bundle()
        query_revision = current.index.revision

        def display_early(evidence):
            with evidence_box.container():
                sources(evidence)

        with answer_column:
            with st.spinner("資料を確認しています…"):
                result = current.engine.answer(
                    question,
                    mode=mode,
                    on_evidence=display_early,
                    audience=audience,
                    detail=detail,
                )
        st.session_state.last_result = result
        st.session_state.last_revision = query_revision
        st.session_state.last_question = question
        st.session_state.last_mode = mode
        st.session_state.last_audience = audience
        st.session_state.last_detail = detail
    result = st.session_state.get("last_result")
    if isinstance(result, LabAnswer):
        with answer_column:
            render_answer(result)
            if (
                mode == "explain"
                and result.status == "answered"
                and result.route == "explain"
            ):
                render_review_actions(result)
            if (
                mode != "explain"
                and result.status == "answered"
                and result.route not in {"baseline", "approved_faq", "explain"}
            ):
                with st.expander("この回答をFAQとして承認"):
                    st.caption(
                        "同じ質問に再利用します。資料の更新・期限切れで無効になります。"
                    )
                    expiry = st.date_input(
                        "有効期限",
                        value=date.today() + timedelta(days=30),
                        min_value=date.today(),
                    )
                    checked = st.checkbox("回答と引用原文を確認しました")
                    if st.button("承認して保存", disabled=not checked):
                        bundle().faq.approve(
                            st.session_state.last_question,
                            result.claims,
                            result.evidence,
                            expires=expiry.isoformat(),
                        )
                        st.success("このPCに保存しました。")
        with evidence_box.container():
            sources(result.evidence)

elif page == "確認済みQ&A":
    st.title("確認済みQ&A")
    st.caption("職員が確認した回答です。元資料が変わったものは再確認が必要です。")
    try:
        reviewed_items = bundle().index.list_reviewed_qa()
    except Exception:
        st.error("一覧を開けませんでした。ローカルAIの接続を確認してください。")
    else:
        if not reviewed_items:
            st.info("保存されたQ&Aはまだありません。")
        status_labels = {
            "confirmed": "確認済み",
            "needs_review": "再確認が必要",
            "archived": "無効",
        }
        audience_labels = {
            "general": "一般向け",
            "child": "子ども向け",
            "staff": "職員向け",
        }
        detail_labels = {"standard": "標準", "short": "短く"}
        for item in reviewed_items:
            with st.container(border=True):
                st.text(item.question)
                st.text(
                    f"{status_labels.get(item.status, item.status)} · 保存日時 {local_time(item.created_at)}"
                )
                with st.expander("回答と根拠"):
                    st.text(
                        f"対象 {audience_labels.get(item.audience, item.audience)} · 長さ {detail_labels.get(item.detail, item.detail)}"
                    )
                    st.text(item.approved_answer)
                    shown_sources = set()
                    for source in item.source_refs:
                        source_key = (source.source_name, source.quote)
                        if source_key in shown_sources:
                            continue
                        shown_sources.add(source_key)
                        st.text(f"根拠資料: {source.source_name}")
                        st.text(source.quote)
                    with st.expander("最初のAI回答"):
                        st.text(item.original_generated_answer)
                    if item.status != "archived":
                        confirm = st.checkbox(
                            "このQ&Aを無効にする",
                            key=f"archive-confirm-{item.id}",
                        )
                        if st.button(
                            "無効にする",
                            key=f"archive-{item.id}",
                            disabled=not confirm,
                        ):
                            try:
                                bundle().index.archive_reviewed_qa(item.id)
                            except Exception:
                                st.error(
                                    "無効にできませんでした。もう一度お試しください。"
                                )
                            else:
                                st.rerun()

elif page == "資料を管理":
    st.title("資料を管理")
    st.caption("使用を許可した資料だけが回答に使われます。")
    if not st.session_state.ready:
        st.info("サイドバーの「AIを準備」を押してください。")
    else:
        current = bundle()
        with st.expander("架空のサンプル資料を追加"):
            if st.button("サンプル5件を登録"):
                with st.spinner("登録しています…"):
                    for path in sorted((ROOT / "sample_docs").glob("*.txt")):
                        current.index.register_bytes(
                            path.name, path.read_bytes(), approved=True
                        )
                st.success("登録しました。")
        upload = st.file_uploader("資料を追加", type=["pdf", "txt", "md"])
        approved = st.checkbox("この資料を回答に使用することを確認しました")
        effective = st.date_input("適用日", value=date.today())
        st.caption(
            "同じ名前で登録すると置き換えます。将来日付の資料は適用日まで検索されません。"
        )
        if st.button(
            "登録する", disabled=upload is None or not approved, type="primary"
        ):
            try:
                with st.spinner("資料を読み込んでいます…"):
                    record = current.index.register_bytes(
                        upload.name,
                        upload.getvalue(),
                        approved=approved,
                        effective_date=effective.isoformat(),
                    )
                st.success("登録しました。")
                st.json(record)
            except Exception as exc:
                st.error(
                    "登録できませんでした。PDFは文字を選択できる形式か確認してください。"
                )
                st.caption(type(exc).__name__)
        for record in current.index.list_documents():
            with st.container(border=True):
                st.text(record["source_name"])
                st.caption(
                    f"{record['page_count']}ページ · {record['chunk_count']}区画 · 適用日 {record.get('effective_date') or '指定なし'}"
                )
                with st.expander("抽出された本文を確認"):
                    for passage in current.index.export_passages():
                        if passage.content_hash == record["content_hash"]:
                            st.text(passage.text)
                original = current.index.get_document(record["content_hash"])
                if original:
                    st.download_button(
                        "登録した原本",
                        original["data"],
                        file_name=original["source_name"],
                        mime=original["mime_type"],
                        key=f"download-{record['source_name']}",
                    )

elif page == "比較結果":
    st.title("比較結果")
    st.caption(
        "架空資料での検証結果です。科学館の実資料での精度を示すものではありません。"
    )
    runs = sorted(
        (ROOT / "outputs" / "lab-comparisons").glob("*/*/report/report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        st.info("まだ比較結果はありません。評価コマンドで作成できます。")
    else:
        selected = st.selectbox(
            "実行記録",
            runs,
            format_func=lambda p: f"{p.parents[2].name} / {p.parents[1].name}",
        )
        report = json.loads(selected.read_text(encoding="utf-8"))
        names = {
            "explain": "わかりやすく説明",
            "baseline": "従来の2段階生成",
            "quoted": "原文から回答",
            "concise": "短く言い換え",
            "bounded": "聞き返し・再検索",
        }

        def fraction(value):
            return f"{value.get('numerator', 0)} / {value.get('denominator', 0)}"

        rows = []
        for mode, values in report.get("modes", {}).items():
            rows.append(
                {
                    "方式": names.get(mode, mode),
                    "実行数": values["attempts"],
                    "回答すべきでない問題への回答": fraction(
                        values["false_answer_rate"]
                    ),
                    "答えられる問題での保留": fraction(values["false_refusal_rate"]),
                    "処理エラー": fraction(values["error_rate"]),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "分母は各条件の問題数です。これは回答内容の正確性を保証する指標ではありません。"
        )
        latency_rows = []
        for mode, values in report.get("modes", {}).items():
            for condition, measures in values.get(
                "latency_by_preparation_exposure_status", {}
            ).items():
                total = measures["total"]
                latency_rows.append(
                    {
                        "方式": names.get(mode, mode),
                        "測定条件": condition,
                        "件数": total["n"],
                        "中央値（秒）": round(total["median_seconds"], 2),
                        "p95（秒）": round(total["p95_seconds"], 2),
                    }
                )
        with st.expander("待ち時間 · 初回／再利用などの条件別"):
            st.dataframe(latency_rows, hide_index=True, width="stretch")
            st.caption(
                "初回のモデル読込と準備後の応答を分けて確認してください。少数例のp95は参考値です。"
            )
        with st.expander("集計の詳細"):
            st.json(report)
    report_path = ROOT / "LAB_RESULTS.md"
    if report_path.exists():
        st.download_button(
            "比較レポート",
            report_path.read_bytes(),
            file_name="LAB_RESULTS.md",
            mime="text/markdown",
        )

else:
    st.title("設定")
    choices = [
        m
        for m in settings.ollama.allowed_models
        if m != settings.ollama.embedding_model
    ]
    selected = st.selectbox(
        "回答モデル", choices, index=choices.index(st.session_state.model)
    )
    if selected != st.session_state.model:
        st.session_state.model = selected
        st.session_state.ready = False
        forget_answer()
        st.rerun()
    st.caption(
        "変更後は「AIを準備」を押してください。モデルの取得は自動では行いません。"
    )
    st.text(f"AI接続先: {settings.ollama.base_url}")
    st.text("質問・回答の利用ログ: 保存しない")
    st.caption(
        "処理時間などの診断ログ: "
        + ("保存する" if settings.logging.enabled else "保存しない")
    )
    st.caption("承認したFAQと登録資料は、この比較版フォルダに保存します。")
    if st.session_state.ready:
        with st.expander("承認済みFAQを整理"):
            confirm = st.checkbox("承認済みFAQをすべて削除する")
            if st.button("FAQを削除", disabled=not confirm):
                bundle().faq.clear()
                st.success("承認済みFAQを削除しました。資料は保持しています。")
