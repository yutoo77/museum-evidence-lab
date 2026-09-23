"""Staff-confirmed Q&A remains tied to active original documents."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from src.lab.contracts import Claim, LabAnswer, SourceCitation
from src.lab.index import HybridIndex


class FakeEmbeddings:
    model_name = "fixture-embedding"

    def __init__(self):
        self.calls = []

    def embed_texts(self, texts, **_kwargs):
        self.calls.append(tuple(texts))
        return [[1.0, 0.0] if "月" in text else [0.0, 1.0] for text in texts]


@pytest.fixture
def prepared(tmp_path):
    provider = FakeEmbeddings()
    index = HybridIndex(
        tmp_path / "index.sqlite3", provider, embedding_digest="sha256:fixture"
    )
    index.register_bytes(
        "月の展示.txt",
        "月の満ち欠けは、太陽光が当たる部分の見え方が変わることで起こります。".encode(),
    )
    passage = index.export_passages()[0]
    claim = Claim(
        "月の満ち欠けは、太陽光の当たり方と見える部分が変わることで起こります。",
        passage.evidence_id,
        passage.text,
        (SourceCitation(passage.evidence_id, passage.text),),
    )
    answer = LabAnswer("answered", "", (claim,), (passage,), route="explain")
    return index, provider, answer


def test_save_list_retrieve_and_archive_without_direct_answer(prepared):
    index, provider, answer = prepared
    before = index.revision
    stored = index.save_reviewed_qa(
        "月の満ち欠けはなぜ起こる？",
        answer,
        "月の形は、太陽に照らされた部分のうち、地球から見える部分が変わるため違って見えます。",
        before,
    )
    assert stored.id > 0
    assert stored.original_generated_answer == answer.text
    assert stored.approved_answer != stored.original_generated_answer
    assert stored.status == "confirmed"
    assert stored.source_evidence_ids == (answer.evidence[0].evidence_id,)
    assert index.list_reviewed_qa() == [stored]
    assert index.revision != before

    call_count = len(provider.calls)
    result = index.search("月の形が日ごとに違う理由は？", include_reviewed_qa=True)
    assert (
        len(provider.calls) == call_count + 1
    )  # One question embedding for both searches.
    assert result.evidence and result.reviewed_qa[0].id == stored.id
    assert result.reviewed_qa[0].approved_answer == stored.approved_answer
    assert index.search("月の形が日ごとに違う理由は？").reviewed_qa == ()

    index.archive_reviewed_qa(stored.id)
    assert index.list_reviewed_qa()[0].status == "archived"
    assert (
        index.search(
            "月の形が日ごとに違う理由は？", include_reviewed_qa=True
        ).reviewed_qa
        == ()
    )


def test_source_update_persistently_requires_review_even_after_reversion(prepared):
    index, _, answer = prepared
    original = answer.evidence[0].text.encode()
    stored = index.save_reviewed_qa(
        "月の形はなぜ変わる？", answer, answer.text, index.revision
    )
    index.register_bytes("月の展示.txt", "月は地球のまわりを回ります。".encode())
    assert index.list_reviewed_qa()[0].status == "needs_review"
    assert (
        index.search("月の形はなぜ変わる？", include_reviewed_qa=True).reviewed_qa == ()
    )
    index.register_bytes("月の展示.txt", original)
    assert index.list_reviewed_qa()[0].id == stored.id
    assert index.list_reviewed_qa()[0].status == "needs_review"
    assert (
        index.search("月の形はなぜ変わる？", include_reviewed_qa=True).reviewed_qa == ()
    )


@pytest.mark.parametrize(
    "registration",
    [
        {"approved": False},
        {
            "approved": True,
            "effective_date": (date.today() + timedelta(days=1)).isoformat(),
        },
    ],
)
def test_source_approval_and_effective_date_changes_disable_reviewed_qa(
    prepared, registration
):
    index, _, answer = prepared
    index.save_reviewed_qa("月の形はなぜ変わる？", answer, answer.text, index.revision)
    index.register_bytes(
        "月の展示.txt", answer.evidence[0].text.encode(), **registration
    )

    assert index.list_reviewed_qa()[0].status == "needs_review"
    assert (
        index.search("月の形はなぜ変わる？", include_reviewed_qa=True).reviewed_qa == ()
    )


def test_multiple_original_sources_are_all_recorded_and_each_can_make_qa_stale(
    tmp_path,
):
    index = HybridIndex(
        tmp_path / "index.sqlite3", FakeEmbeddings(), embedding_digest="sha256:fixture"
    )
    index.register_bytes(
        "しくみ.txt", "月の形は太陽光が当たる部分の見え方で変わります。".encode()
    )
    index.register_bytes(
        "体験.txt", "模型では月と地球の位置を変えて見え方を比べます。".encode()
    )
    passages = index.export_passages()
    references = tuple(SourceCitation(item.evidence_id, item.text) for item in passages)
    answer = LabAnswer(
        "answered",
        "",
        (
            Claim(
                "月の形の見え方を模型で比べます。",
                references[0].evidence_id,
                references[0].quote,
                references,
            ),
        ),
        passages,
        route="explain",
    )
    stored = index.save_reviewed_qa(
        "月の形を模型でどう比べる？", answer, answer.text, index.revision
    )
    assert {ref.source_name for ref in stored.source_refs} == {"しくみ.txt", "体験.txt"}
    assert len(index.list_reviewed_qa()[0].source_refs) == 2
    index.register_bytes("体験.txt", "模型では月の位置を変えて観察します。".encode())
    assert index.list_reviewed_qa()[0].status == "needs_review"


def test_reviewed_qa_is_scoped_to_audience_and_detail(prepared):
    index, _, answer = prepared
    stored = index.save_reviewed_qa(
        "月の満ち欠けはなぜ起こる？",
        answer,
        answer.text,
        index.revision,
        audience="child",
        detail="short",
    )
    assert (
        index.search("月の形が変わる理由は？", include_reviewed_qa=True).reviewed_qa
        == ()
    )
    matched = index.search(
        "月の形が変わる理由は？",
        include_reviewed_qa=True,
        reviewed_audience="child",
        reviewed_detail="short",
    )
    assert matched.reviewed_qa[0].id == stored.id
    assert (
        index.search(
            "火星の温度は？",
            include_reviewed_qa=True,
            reviewed_audience="child",
            reviewed_detail="short",
        ).reviewed_qa
        == ()
    )


def test_save_rejects_old_revision_and_unverified_or_oversized_content(prepared):
    index, _, answer = prepared
    with pytest.raises(ValueError, match="質問・確認済み回答"):
        index.save_reviewed_qa("質問", answer, " ", index.revision)
    with pytest.raises(ValueError, match="質問・確認済み回答"):
        index.save_reviewed_qa("質問", answer, "長" * 804, index.revision)
    with pytest.raises(ValueError, match="数値"):
        index.save_reviewed_qa(
            "月の質問", answer, "月の体験は35分です。", index.revision
        )
    with pytest.raises(ValueError, match="更新"):
        index.save_reviewed_qa("月の質問", answer, answer.text, "wrong")
    index.register_bytes("月の展示.txt", "月は地球の衛星です。".encode())
    with pytest.raises(ValueError, match="更新"):
        index.save_reviewed_qa("月の質問", answer, answer.text, index.revision)
    assert index.list_reviewed_qa() == []


def test_existing_v1_index_is_opened_without_rebuilding_documents(prepared):
    index, provider, _ = prepared
    original = index.list_documents()
    with sqlite3.connect(index.db_path) as db:
        db.execute("DROP TABLE reviewed_qa_sources")
        db.execute("DROP TABLE reviewed_qa")
    reopened = HybridIndex(index.db_path, provider, embedding_digest="sha256:fixture")
    assert reopened.list_documents() == original
    assert reopened.search("月の形は？").evidence
    assert reopened.list_reviewed_qa() == []
