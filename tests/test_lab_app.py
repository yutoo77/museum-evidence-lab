from dataclasses import replace
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.lab import factory
from src.lab.contracts import Claim, Evidence, LabAnswer, SourceCitation


@pytest.mark.parametrize("page", ["資料を調べる", "資料を管理", "比較結果", "設定"])
def test_lab_starts_without_inference_on_each_page(page):
    app = AppTest.from_file("lab_app.py", default_timeout=30).run()
    app.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value == page
    assert app.session_state["ready"] is False


@pytest.fixture
def fake_lab(monkeypatch):
    evidence = (
        Evidence("moon", "月は太陽の光を反射します。", "月の資料.pdf", 2, "moon-hash"),
        Evidence(
            "phase",
            "地球から見える明るい部分が変わります。",
            "満ち欠けの資料.pdf",
            5,
            "phase-hash",
        ),
    )
    explanation = "月の満ち欠けでは、太陽に照らされた部分の見え方が変わります。"
    citations = tuple(SourceCitation(item.evidence_id, item.text) for item in evidence)
    result = LabAnswer(
        "answered",
        "",
        claims=(Claim(explanation, "moon", evidence[0].text, citations=citations),),
        evidence=evidence,
        route="explain",
    )

    class FakeEngine:
        def __init__(self):
            self.calls = []
            self.result = result

        def answer(self, question, mode="explain", on_evidence=None, **options):
            self.calls.append({"question": question, "mode": mode, **options})
            if on_evidence:
                on_evidence(self.result.evidence)
            return replace(self.result, route=mode)

    class FakeFAQ:
        def __init__(self):
            self.approved = []

        def approve(self, *args, **kwargs):
            self.approved.append((args, kwargs))

    current = SimpleNamespace(engine=FakeEngine(), faq=FakeFAQ())
    monkeypatch.setattr(factory, "build_lab", lambda *args, **kwargs: current)
    st.cache_resource.clear()
    yield current
    st.cache_resource.clear()


def ready_app():
    app = AppTest.from_file("lab_app.py", default_timeout=30)
    app.session_state["ready"] = True
    app.run()
    app.text_area[0].set_value("月の満ち欠けはなぜ起こる？").run()
    return app


def ask(app):
    next(button for button in app.button if button.label == "調べる").click().run()
    assert not app.exception


def direct_text(block):
    return [item.value for item in block.children.values() if item.type == "text"]


def test_lab_explanation_ui_defaults_to_general_standard_explanation(fake_lab):
    app = ready_app()
    assert app.selectbox(key="answer_mode").value == "わかりやすく説明"
    assert app.selectbox(key="audience").value == "一般向け"
    assert app.selectbox(key="detail").value == "標準"
    ask(app)
    assert fake_lab.engine.calls == [
        {
            "question": "月の満ち欠けはなぜ起こる？",
            "mode": "explain",
            "audience": "general",
            "detail": "standard",
        }
    ]
    assert app.subheader[0].value == "説明案"
    assert "利用前に根拠をご確認ください。" in [item.value for item in app.caption]
    assert fake_lab.engine.result.claims[0].text in [item.value for item in app.text]


def test_lab_explanation_ui_all_citations_are_collapsed_with_source_pages(fake_lab):
    app = ready_app()
    ask(app)
    citations = next(item for item in app.expander if item.label == "1. この説明の根拠")
    assert citations.proto.expanded is False
    assert direct_text(citations) == [
        "出典: 月の資料.pdf / 2ページ",
        "月は太陽の光を反射します。",
        "出典: 満ち欠けの資料.pdf / 5ページ",
        "地球から見える明るい部分が変わります。",
    ]
    passages = next(item for item in app.expander if item.label == "検索した原文")
    assert passages.proto.expanded is False
    assert "根拠の原文" not in [item.value for item in app.subheader]
    assert "この回答をFAQとして承認" not in [item.label for item in app.expander]
    assert not fake_lab.faq.approved


def test_lab_explanation_ui_each_paragraph_keeps_its_own_references(fake_lab):
    first = fake_lab.engine.result.claims[0]
    second_source = replace(fake_lab.engine.result.evidence[1], page_number=None)
    second = Claim("明るい部分の見え方が変わります。", "phase", second_source.text)
    fake_lab.engine.result = replace(
        fake_lab.engine.result,
        claims=(first, second),
        evidence=(fake_lab.engine.result.evidence[0], second_source),
    )
    app = ready_app()
    ask(app)
    citations = next(item for item in app.expander if item.label == "2. この説明の根拠")
    assert direct_text(citations) == [
        "出典: 満ち欠けの資料.pdf",
        "地球から見える明るい部分が変わります。",
    ]
    assert first.text in [item.value for item in app.text]
    assert second.text in [item.value for item in app.text]


@pytest.mark.parametrize("mode", ["explain", "concise"])
def test_lab_explanation_ui_context_keeps_subject_once_per_cited_passage(
    fake_lab, mode
):
    workshop = Evidence(
        "workshop",
        "地学ワークショップ\n所要時間は35分です。参加費は500円です。",
        "体験教室.pdf",
        7,
        "workshop-hash",
    )
    reservation = Evidence(
        "reservation",
        "地学ワークショップの予約\n開始前に受付してください。",
        "受付案内.pdf",
        2,
        "reservation-hash",
    )
    uncited = Evidence(
        "garden", "温室は自由に見学できます。", "温室.txt", None, "garden-hash"
    )
    references = (
        SourceCitation("workshop", "所要時間は35分です。"),
        SourceCitation("workshop", "参加費は500円です。"),
        SourceCitation("reservation", "開始前に受付してください。"),
    )
    fake_lab.engine.result = replace(
        fake_lab.engine.result,
        evidence=(workshop, reservation, uncited),
        claims=(
            Claim(
                "地学ワークショップの案内です。",
                "workshop",
                references[0].quote,
                references,
            ),
        ),
    )
    app = ready_app()
    if mode == "concise":
        app.selectbox(key="answer_mode").set_value("比較用（試験）").run()
    ask(app)
    citations = next(item for item in app.expander if item.label == "1. この説明の根拠")
    contexts = [item for item in citations.children.values() if item.type == "expander"]
    assert [item.label for item in contexts] == ["周辺の原文 1", "周辺の原文 2"]
    assert all(item.proto.expanded is False for item in contexts)
    assert [item.value for item in contexts[0].text] == [
        "出典: 体験教室.pdf / 7ページ",
        workshop.text,
    ]
    assert [item.value for item in contexts[1].text] == [
        "出典: 受付案内.pdf / 2ページ",
        reservation.text,
    ]
    assert all(reference.quote in direct_text(citations) for reference in references)
    assert workshop.text not in direct_text(citations)
    assert reservation.text not in direct_text(citations)
    assert uncited.text not in [item.value for item in citations.text]
    if mode == "explain":
        passages = next(item for item in app.expander if item.label == "検索した原文")
        assert uncited.text in [item.value for item in passages.text]


@pytest.mark.parametrize(
    ("key", "label", "expected"),
    [
        ("audience", "子ども向け", "child"),
        ("audience", "職員向け", "staff"),
        ("detail", "短く", "short"),
    ],
)
def test_lab_explanation_ui_option_change_clears_old_answer_and_is_forwarded(
    fake_lab, key, label, expected
):
    app = ready_app()
    ask(app)
    app.selectbox(key=key).set_value(label).run()
    assert not app.exception
    assert "last_result" not in app.session_state
    assert fake_lab.engine.result.claims[0].text not in [
        item.value for item in app.text
    ]
    ask(app)
    assert fake_lab.engine.calls[-1][key] == expected


@pytest.mark.parametrize("change", ["question", "mode"])
def test_lab_explanation_ui_question_or_mode_change_clears_old_answer(fake_lab, change):
    app = ready_app()
    ask(app)
    if change == "question":
        app.text_area[0].set_value("月はどうして明るい？").run()
    else:
        app.selectbox(key="answer_mode").set_value("原文から回答").run()
    assert not app.exception
    assert "last_result" not in app.session_state


def test_lab_explanation_ui_quoted_answer_keeps_legacy_source_and_faq(fake_lab):
    evidence = fake_lab.engine.result.evidence[0]
    fake_lab.engine.result = replace(
        fake_lab.engine.result,
        claims=(Claim(evidence.text, evidence.evidence_id, evidence.text),),
        evidence=(evidence,),
    )
    app = ready_app()
    app.selectbox(key="answer_mode").set_value("原文から回答").run()
    assert app.selectbox(key="audience").disabled
    assert app.selectbox(key="detail").disabled
    ask(app)
    assert fake_lab.engine.calls[-1]["mode"] == "quoted"
    assert "出典: 月の資料.pdf / 2ページ" in [item.value for item in app.text]
    assert "根拠の原文" in [item.value for item in app.subheader]
    assert "利用前に根拠をご確認ください。" not in [item.value for item in app.caption]
    assert "この回答をFAQとして承認" in [item.label for item in app.expander]
    next(
        item for item in app.checkbox if item.label == "回答と引用原文を確認しました"
    ).check().run()
    next(item for item in app.button if item.label == "承認して保存").click().run()
    assert not app.exception
    assert len(fake_lab.faq.approved) == 1


@pytest.mark.parametrize(
    ("label", "mode"),
    [
        ("短く言い換え（試験）", "concise"),
        ("聞き返し・再検索（試験）", "bounded"),
        ("従来の2段階生成（比較用）", "baseline"),
    ],
)
def test_lab_explanation_ui_existing_comparison_modes_remain_accessible(
    fake_lab, label, mode
):
    app = ready_app()
    app.selectbox(key="answer_mode").set_value("比較用（試験）").run()
    next(item for item in app.selectbox if item.label == "比較方式").set_value(
        label
    ).run()
    ask(app)
    assert fake_lab.engine.calls[-1]["mode"] == mode
    assert "利用前に根拠をご確認ください。" not in [item.value for item in app.caption]


def test_lab_explanation_ui_never_renders_answer_or_citation_as_markup(fake_lab):
    unsafe = '![remote](https://example.invalid/image.png)<script>alert("x")</script>'
    evidence = replace(
        fake_lab.engine.result.evidence[0], source_name=unsafe, text=unsafe
    )
    fake_lab.engine.result = replace(
        fake_lab.engine.result,
        evidence=(evidence,),
        claims=(Claim(unsafe, evidence.evidence_id, unsafe),),
    )
    app = ready_app()
    ask(app)
    assert unsafe in [item.value for item in app.text]
    assert all(unsafe not in item.value for item in app.markdown)


def test_lab_explanation_ui_refusal_does_not_offer_faq(fake_lab):
    fake_lab.engine.result = LabAnswer(
        "refused", "資料だけでは確認できません。", route="explain"
    )
    app = ready_app()
    ask(app)
    assert "資料だけでは確認できません。" in [item.value for item in app.text]
    assert "この回答をFAQとして承認" not in [item.label for item in app.expander]
