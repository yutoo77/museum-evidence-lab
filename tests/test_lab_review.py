"""Independent integration regressions for local-only, bounded evidence answers."""

from __future__ import annotations

import json
import socket
from datetime import date, timedelta

import httpx
import pytest

from src.lab.audit import AuditLog
from src.lab.client import InstrumentedOllamaClient, SafeEmbeddingClient
from src.lab.contracts import Claim, Evidence, GenerationResult, SearchResult
from src.lab.engine import LabEngine, source_sentences, validate_selection
from src.lab.faq import ApprovedFAQ
from src.lab.index import HybridIndex, IndexMismatchError

_MODEL = "qwen3:1.7b"
_DIGEST = "a" * 64
_EMBEDDING_DIGEST = "b" * 64
_PASSAGE = Evidence(
    "moon-1", "月は太陽の光を反射します。", "moon.txt", None, "content-1", 0.1
)


class _Index:
    def __init__(self, evidence=(_PASSAGE,)):
        self.evidence = evidence
        self.calls = []

    def search(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return SearchResult(self.evidence, self.evidence)

    def export_passages(self):
        return self.evidence


class _Client:
    model_name = _MODEL

    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return GenerationResult('{"selection":["P1S1"]}', "stop")


class _MutableEmbedding:
    model_name = "embeddinggemma"

    def __init__(self, *args, **kwargs):
        self.model_digest = _EMBEDDING_DIGEST
        self.last_metrics = {}

    def embed_texts(self, texts):
        self.last_metrics = {"digest": self.model_digest}
        return [[1.0, 0.0] for _ in texts]

    def close(self):
        pass


@pytest.mark.parametrize(
    "value",
    [
        {"status": [], "claims": []},
        {"status": {"private": "text"}, "claims": []},
        {"status": "answered", "claims": [{"sentence_id": []}]},
        {"status": "answered", "claims": [{"sentence_id": {}}]},
        {"status": "answered", "claims": {}},
    ],
)
def test_malformed_json_shapes_fail_closed_without_exposing_content(value):
    answer = validate_selection(json.dumps(value), (_PASSAGE,), concise=False)
    assert answer.status == "needs_review"
    assert not answer.claims
    assert "private" not in " ".join(answer.issues)


def test_new_numeric_substrings_are_not_accepted_as_source_numbers():
    passage = Evidence(
        "number-1", "測定値は110付近です。", "number.txt", None, "number-hash"
    )
    answer = validate_selection(
        json.dumps(
            {
                "status": "answered",
                "claims": [{"sentence_id": "P1S1", "text": "値は10です。"}],
            }
        ),
        (passage,),
        concise=True,
    )
    assert answer.status == "needs_review"
    assert "unsupported_number" in answer.issues


def test_valid_short_japanese_answer_is_available_for_selection():
    passage = Evidence("fee-1", "無料です。", "fee.txt", None, "fee-hash")
    assert any(
        text == "無料です。" for _, text in source_sentences((passage,)).values()
    )


def test_optional_audit_failure_does_not_destroy_valid_answer():
    class BrokenAudit:
        def save(self, answer):
            raise OSError("private endpoint payload must not escape")

    answer = LabEngine(_Index(), _Client(), audit=BrokenAudit()).answer(
        "月はなぜ光りますか", mode="quoted"
    )
    assert answer.status == "answered"
    assert answer.text == _PASSAGE.text
    assert "private endpoint payload" not in " ".join(answer.issues)


def test_expired_total_budget_stops_before_retrieval():
    index = _Index()
    client = _Client()
    answer = LabEngine(index, client, budget_seconds=0).answer(
        "月はなぜ光りますか", mode="quoted"
    )
    assert answer.status == "needs_review"
    assert not index.calls
    assert not client.calls


def test_source_changed_during_selection_never_returns_stale_answer():
    class VersionedIndex(_Index):
        revision = "before-update"

        def search(self, question, **kwargs):
            return SearchResult(self.evidence, self.evidence, revision=self.revision)

    index = VersionedIndex()

    class UpdatingClient(_Client):
        def chat(self, *args, **kwargs):
            index.revision = "after-update"
            return super().chat(*args, **kwargs)

    answer = LabEngine(index, UpdatingClient()).answer(
        "月はなぜ光りますか", mode="quoted"
    )
    assert answer.status != "answered"
    assert not answer.claims


@pytest.mark.parametrize("operation", ["search", "register"])
def test_same_tag_new_weights_are_rejected_by_long_lived_index(tmp_path, operation):
    provider = _MutableEmbedding()
    index = HybridIndex(
        tmp_path / "index.sqlite3", provider, embedding_digest=provider.model_digest
    )
    index.register_bytes("moon.txt", _PASSAGE.text.encode())
    provider.model_digest = "c" * 64
    with pytest.raises(IndexMismatchError):
        if operation == "search":
            index.search("月はなぜ光りますか")
        else:
            index.register_bytes("moon.txt", "月の別の資料です。".encode())


def test_factory_uses_the_configuration_overlap_contract(tmp_path, monkeypatch):
    from src.lab import factory

    class FakeGeneration:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(factory, "SafeEmbeddingClient", _MutableEmbedding)
    monkeypatch.setattr(factory, "InstrumentedOllamaClient", FakeGeneration)
    config = tmp_path / "lab.yaml"
    config.write_text(
        "storage:\n  root: data/lab\nretrieval:\n  overlap: 100\n", encoding="utf-8"
    )
    bundle = factory.build_lab(config)
    try:
        assert bundle.index.chunk_overlap == 100
    finally:
        bundle.close()


def test_approved_faq_is_invalidated_by_same_name_source_update(tmp_path):
    provider = _MutableEmbedding()
    index = HybridIndex(
        tmp_path / "index.sqlite3", provider, embedding_digest=provider.model_digest
    )
    index.register_bytes("moon.txt", _PASSAGE.text.encode())
    passage = index.export_passages()[0]
    faq = ApprovedFAQ(tmp_path / "faq.sqlite3")
    claim = Claim(passage.text, passage.evidence_id, passage.text)
    question = "月はなぜ光りますか"
    faq.approve(
        question,
        (claim,),
        (passage,),
        expires=(date.today() + timedelta(days=7)).isoformat(),
    )
    assert faq.lookup(question, index.export_passages()) is not None
    index.register_bytes("moon.txt", "月は太陽光を反射して見えます。".encode())
    assert faq.lookup(question, index.export_passages()) is None


def test_audit_metadata_never_contains_question_answer_quote_or_model_thought(tmp_path):
    audit = AuditLog(tmp_path / "audit", enabled=True)
    answer = LabEngine(_Index(), _Client(), audit=audit).answer(
        "PRIVATE_QUESTION_123", mode="quoted"
    )
    text = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "audit").glob("*.jsonl")
    )
    assert answer.status == "answered"
    assert "PRIVATE_QUESTION_123" not in text
    assert _PASSAGE.text not in text
    assert "moon.txt" not in text
    assert "content-1" not in text


def test_full_mocked_pipeline_does_not_open_sockets_with_hostile_proxy_env(
    tmp_path, monkeypatch
):
    """An offline simulation checks app egress, not Ollama's separate process."""
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append((args, kwargs))
        raise AssertionError("network access is forbidden by this test")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("HTTPS_PROXY", "http://external.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://external.invalid:8080")
    calls = []

    def handle(request):
        calls.append(str(request.url))
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": _MODEL, "digest": _DIGEST},
                        {"name": "embeddinggemma:latest", "digest": _EMBEDDING_DIGEST},
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "details": {"format": "gguf"},
                    "model_info": {"general.architecture": "local"},
                },
            )
        if request.url.path == "/api/embed":
            count = len(json.loads(request.content)["input"])
            return httpx.Response(
                200,
                json={
                    "model": "embeddinggemma:latest",
                    "embeddings": [[1, 0] for _ in range(count)],
                },
            )
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "model": _MODEL,
                        "done": True,
                        "done_reason": "stop",
                        "message": {
                            "content": '{"selection":["P1S1"]}',
                            "thinking": "PRIVATE_THOUGHT",
                        },
                    }
                )
                + "\n",
            )
        raise AssertionError("Unexpected endpoint")

    models = [_MODEL, "embeddinggemma"]
    with SafeEmbeddingClient(
        "http://localhost:11435",
        "embeddinggemma",
        models,
        transport=httpx.MockTransport(handle),
    ) as embedding:
        with InstrumentedOllamaClient(
            "http://localhost:11435",
            _MODEL,
            models,
            transport=httpx.MockTransport(handle),
        ) as client:
            index = HybridIndex(
                tmp_path / "index.sqlite3",
                embedding,
                embedding_digest=_EMBEDDING_DIGEST,
            )
            index.register_bytes("moon.txt", _PASSAGE.text.encode())
            answer = LabEngine(index, client).answer(
                "月はなぜ光りますか", mode="quoted"
            )
    assert answer.status == "answered"
    assert "PRIVATE_THOUGHT" not in str(answer)
    assert calls
    assert not attempts
