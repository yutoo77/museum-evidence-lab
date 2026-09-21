"""Offline retrieval/versioning regression tests with deterministic vectors."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from src.exceptions import UnsupportedDocumentError
from src.lab.index import HybridIndex, IndexMismatchError, InvalidEmbeddingError


class FakeEmbeddings:
    model_name = "test-embedding"

    def __init__(self):
        self.calls = []
        self.override = None

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if self.override is not None:
            return [self.override for _ in texts]
        vectors = []
        for text in texts:
            if "月" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "火星" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


@pytest.fixture
def fixture_index(tmp_path):
    provider = FakeEmbeddings()
    index = HybridIndex(
        tmp_path / "local" / "index.sqlite3", provider, embedding_digest="sha256:test"
    )
    return index, provider


def test_register_search_download_and_safe_name(fixture_index):
    index, provider = fixture_index
    text = "月は地球の衛星です。月の形の変化は太陽光が当たる部分の見え方によります。"
    item = index.register_bytes("../../月.txt", text.encode())
    result = index.search("月は何ですか？")
    assert item["source_name"] == "月.txt"
    assert result.evidence[0].text == text
    assert result.evidence[0].content_hash == item["content_hash"]
    assert result.evidence[0].version_id == item["content_hash"]
    assert result.evidence[0].evidence_id.startswith("E")
    assert result.revision == index.revision
    assert index.get_document(item["content_hash"])["data"] == text.encode()
    assert index.get_document("not-a-hash") is None
    assert len(provider.calls) == 2
    assert not (index.db_path.parent.parent / "月.txt").exists()


def test_same_name_is_atomic_replacement_and_changes_revision(fixture_index):
    index, provider = fixture_index
    old = index.register_bytes("Moon.TXT", "月の旧資料です。".encode())
    revision = index.revision
    old_ids = {item.evidence_id for item in index.export_passages()}
    new = index.register_bytes("moon.txt", "月の新資料です。".encode())
    assert len(index.list_documents()) == 1
    assert index.revision != revision
    assert index.get_document(old["content_hash"]) is None
    assert index.get_document(new["content_hash"]) is not None
    assert old_ids.isdisjoint(item.evidence_id for item in index.export_passages())
    count = len(provider.calls)
    assert index.register_bytes("moon.txt", "月の新資料です。".encode()) == new
    assert len(provider.calls) == count


def test_failed_replacement_retains_original(fixture_index):
    index, provider = fixture_index
    original = index.register_bytes("月.txt", "月の元の資料です。".encode())
    revision = index.revision
    provider.override = [float("nan"), 0, 1]
    with pytest.raises(InvalidEmbeddingError):
        index.register_bytes("月.txt", "月の更新資料です。".encode())
    assert index.revision == revision
    assert index.list_documents() == [original]


@pytest.mark.parametrize("change", ["digest", "name", "chunks"])
def test_fingerprint_mismatch_refuses_before_query_embedding(fixture_index, change):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は地球の衛星です。".encode())
    before = len(provider.calls)
    digest = "sha256:other" if change == "digest" else "sha256:test"
    if change == "name":
        provider.model_name = "changed-name"
    other = HybridIndex(
        index.db_path,
        provider,
        embedding_digest=digest,
        chunk_size=600 if change == "chunks" else 800,
    )
    with pytest.raises(IndexMismatchError):
        other.search("月とは？")
    assert len(provider.calls) == before


def test_embedding_dimension_change_is_detected(fixture_index):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は衛星です。".encode())
    provider.override = [1, 0]
    with pytest.raises(IndexMismatchError):
        index.search("月とは？")
    with pytest.raises(IndexMismatchError):
        index.register_bytes("火星.txt", "火星は惑星です。".encode())
    assert len(index.list_documents()) == 1


@pytest.mark.parametrize(
    "invalid", [[], [0, 0, 0], [float("inf"), 1, 0], [float("nan"), 1, 0]]
)
def test_invalid_embeddings_do_not_write_partial_document(fixture_index, invalid):
    index, provider = fixture_index
    provider.override = invalid
    with pytest.raises(InvalidEmbeddingError):
        index.register_bytes("資料.txt", "数値に変換できない資料です。".encode())
    assert index.list_documents() == []


def test_bad_embedding_count_is_rejected(fixture_index):
    index, provider = fixture_index
    provider.embed_texts = lambda texts: []
    with pytest.raises(InvalidEmbeddingError):
        index.register_bytes("資料.txt", "登録テストです。".encode())
    assert index.list_documents() == []


def test_empty_and_unsupported_documents(fixture_index):
    index, provider = fixture_index
    assert index.search("質問").evidence == ()
    assert index.export_passages() == ()
    assert provider.calls == []
    with pytest.raises(UnsupportedDocumentError):
        index.register_bytes("危険.html", b"<script>nothing</script>")
    assert provider.calls == []


def test_approval_effective_date_and_cache_invalidation(fixture_index):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は衛星です。".encode(), approved=False)
    index.register_bytes(
        "火星.txt",
        "火星は惑星です。".encode(),
        effective_date=date.today() + timedelta(days=30),
    )
    assert index.export_passages() == ()
    assert index.search("月とは？").evidence == ()
    revision = index.revision
    index.register_bytes("月.txt", "月は衛星です。".encode(), approved=True)
    assert index.revision != revision
    assert len(index.export_passages()) == 1
    assert index.search("月とは？").evidence[0].source_name == "月.txt"


def test_lexical_japanese_exhibit_name_and_number_without_model_calls(fixture_index):
    index, provider = fixture_index
    index.register_bytes(
        "展示.txt", "展示番号M87のカムイ分光器は光を波長ごとに分ける装置です。".encode()
    )
    index.register_bytes("月.txt", "月は地球の衛星です。".encode())
    before = len(provider.calls)
    result = index.search("M87のカムイ分光器について教えてください", mode="lexical")
    assert [item.source_name for item in result.evidence] == ["展示.txt"]
    assert result.evidence[0].lexical_score > 0
    assert result.embedding_seconds == 0
    assert len(provider.calls) == before


def test_hybrid_lexical_can_rescue_exact_match_but_not_unrelated(fixture_index):
    index, provider = fixture_index
    index.register_bytes("展示.txt", "M87のカムイ分光器は光を分ける装置です。".encode())
    index.register_bytes("火星.txt", "火星は赤い惑星です。".encode())
    provider.override = [1, 0, 0]
    result = index.search("M87のカムイ分光器とは？")
    assert [item.source_name for item in result.evidence] == ["展示.txt"]
    assert result.evidence[0].distance == 1
    assert index.search("量子コンピュータとは何ですか？").evidence == ()
    assert index.search("とは何ですか？", mode="lexical").evidence == ()


def test_full_chunk_budget_and_duplicate_text(fixture_index):
    index, provider = fixture_index
    short = "月は衛星です。"
    long = "月の資料は長文です。" * 30
    index.register_bytes("短文.txt", short.encode())
    index.register_bytes("複製.txt", short.encode())
    index.register_bytes("長文.txt", long.encode())
    result = index.search("月について", max_context_chars=len(short))
    assert len(result.evidence) == 1
    assert result.evidence[0].text == short
    assert len(result.candidates) == 3
    assert index.search("月について", max_context_chars=len(short) - 1).evidence == ()


def test_dense_reference_keeps_unscreened_top_five(fixture_index):
    index, provider = fixture_index
    for number in range(6):
        index.register_bytes(f"資料{number}.txt", f"火星の資料{number}です。".encode())
    provider.override = [1, 0, 0]
    result = index.search("月の質問", mode="dense", max_evidence=1, max_context_chars=1)
    assert len(result.evidence) == 5
    assert len(result.candidates) == 6
    assert all(item.distance == 1 for item in result.evidence)


def test_corrupt_vector_is_rejected_instead_of_returning_silent_ranking(fixture_index):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は衛星です。".encode())
    with sqlite3.connect(index.db_path) as db:
        db.execute("UPDATE passages SET vector = '[0,0,0]'")
    with pytest.raises(InvalidEmbeddingError):
        index.search("月とは？")


def test_invalid_approval_or_date_rejected_before_embeddings(fixture_index):
    index, provider = fixture_index
    with pytest.raises(ValueError):
        index.register_bytes("資料.txt", b"text", approved="false")
    with pytest.raises(ValueError):
        index.register_bytes("資料.txt", b"text", effective_date="yesterday")
    assert provider.calls == []


def test_document_byte_limit_is_checked_before_loading(fixture_index, monkeypatch):
    index, provider = fixture_index
    monkeypatch.setattr("src.lab.index.MAX_DOCUMENT_BYTES", 4)
    with pytest.raises(ValueError, match="50 MB"):
        index.register_bytes("資料.txt", b"12345")
    assert provider.calls == []
    assert index.list_documents() == []


def test_relevance_margin_filters_weak_dense_passages(fixture_index):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は地球の衛星です。".encode())
    provider.override = [0.6, 0.8, 0]
    index.register_bytes("弱い候補.txt", "別の内容です。".encode())
    provider.override = [1, 0, 0]
    result = index.search("月の解説")
    assert len(result.candidates) == 2
    assert [item.source_name for item in result.evidence] == ["月.txt"]


def test_sql_failure_rolls_back_replacement_and_metadata(fixture_index):
    index, provider = fixture_index
    original = index.register_bytes("月.txt", "月の元の資料です。".encode())
    revision = index.revision
    with sqlite3.connect(index.db_path) as db:
        db.execute(
            """CREATE TRIGGER test_reject_insert BEFORE INSERT ON passages
            BEGIN SELECT RAISE(ABORT, 'insertion failed'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError):
        index.register_bytes("月.txt", "月の更新資料です。".encode())
    assert index.revision == revision
    assert index.list_documents() == [original]
    assert index.get_document(original["content_hash"]) is not None


@pytest.mark.parametrize("operation", ["search", "register"])
def test_same_name_changed_weights_rejected_in_long_lived_index(
    fixture_index, operation
):
    index, provider = fixture_index
    provider.last_metrics = {"digest": "sha256:test"}
    original = index.register_bytes("月.txt", "月の元の資料です。".encode())
    provider.last_metrics = {"digest": "sha256:changed-weights"}
    with pytest.raises(IndexMismatchError):
        if operation == "search":
            index.search("月について")
        else:
            index.register_bytes("月.txt", "月の更新資料です。".encode())
    assert index.list_documents() == [original]


def test_search_passes_remaining_budget_and_checks_after_embedding(
    fixture_index, monkeypatch
):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は衛星です。".encode())
    clock = [100.0]
    monkeypatch.setattr("src.lab.index.time.perf_counter", lambda: clock[0])
    seen = []

    def slow_embedding(texts, *, timeout_seconds):
        seen.append(timeout_seconds)
        clock[0] += 2
        return [[1, 0, 0]]

    provider.embed_texts = slow_embedding
    with pytest.raises(TimeoutError):
        index.search("月について", timeout_seconds=1)
    assert seen == [1]


def test_search_checks_deadline_after_local_snapshot(fixture_index, monkeypatch):
    index, provider = fixture_index
    index.register_bytes("月.txt", "月は衛星です。".encode())
    clock = [100.0]
    monkeypatch.setattr("src.lab.index.time.perf_counter", lambda: clock[0])
    original_snapshot = index._snapshot

    def slow_snapshot():
        result = original_snapshot()
        clock[0] += 2
        return result

    monkeypatch.setattr(index, "_snapshot", slow_snapshot)
    calls_before = len(provider.calls)
    with pytest.raises(TimeoutError):
        index.search("月について", timeout_seconds=1)
    assert len(provider.calls) == calls_before
