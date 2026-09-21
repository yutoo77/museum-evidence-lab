"""Local-model boundaries and response measurements without a running Ollama."""

from __future__ import annotations

import json

import httpx
import pytest

from src.lab.client import (
    InstrumentedOllamaClient,
    LocalModelError,
    LocalModelSecurityError,
    SafeEmbeddingClient,
    inspect_health,
    validate_base_url,
)

DIGEST = "a" * 64
MODEL = "qwen3:1.7b"


def local_metadata():
    return {
        "details": {"format": "gguf"},
        "model_info": {"general.architecture": "qwen3"},
    }


def chunk(content="", *, done=False, **extra):
    return {"model": MODEL, "message": {"content": content}, "done": done, **extra}


class OllamaStub:
    def __init__(self):
        self.calls = []
        self.digest = DIGEST
        self.metadata = local_metadata()
        self.stream = [
            chunk("月は", done=False),
            chunk(
                "自ら光りません。",
                done=True,
                done_reason="stop",
                total_duration=2_000_000_000,
                load_duration=250_000_000,
                prompt_eval_duration=500_000_000,
                eval_duration=1_250_000_000,
                prompt_eval_count=30,
                eval_count=8,
            ),
        ]

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": MODEL, "digest": self.digest},
                        {"name": "embeddinggemma:latest", "digest": "b" * 64},
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json=self.metadata)
        if request.url.path == "/api/chat":
            return httpx.Response(
                200, text="\n".join(json.dumps(item) for item in self.stream)
            )
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                json={
                    "model": "embeddinggemma:latest",
                    "embeddings": [[1.0, 0.0] for _ in body["input"]],
                },
            )
        if request.url.path == "/api/generate":
            return httpx.Response(
                200, json={"model": MODEL, "done": True, "done_reason": "load"}
            )
        raise AssertionError(request.url)

    def client(self):
        return InstrumentedOllamaClient(
            "http://localhost:11435",
            MODEL,
            [MODEL, "embeddinggemma"],
            transport=httpx.MockTransport(self.handle),
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://192.168.1.2:11434",
        "http://127.0.0.1.evil.test",
        "http://username@localhost:11435",
        "http://localhost:11435/api",
        "http://localhost:11435?",
        "http://localhost:11435#fragment",
        "http://localhost\\@example.com",
        "file:///tmp/model",
        "http://127.0.0.1:0",
        "http://[::ffff:192.168.1.1]",
        "http://0.0.0.0",
        "http://127.1",
    ],
)
def test_rejects_non_loopback_or_ambiguous_endpoints(url):
    with pytest.raises(LocalModelSecurityError):
        validate_base_url(url)


def test_canonicalizes_localhost_without_dns_and_supports_ipv6():
    assert validate_base_url("http://localhost:11435/") == "http://127.0.0.1:11435"
    assert validate_base_url("http://[::1]:11435") == "http://[::1]:11435"


def test_stream_collects_only_final_content_and_honest_measurements():
    stub = OllamaStub()
    stub.stream[0]["message"]["thinking"] = "secret internal reasoning"
    with stub.client() as client:
        result = client.chat(
            "instruction", "question", max_tokens=120, schema={"type": "object"}
        )
    assert result.content == "月は自ら光りません。"
    assert result.done_reason == "stop"
    assert result.metrics["total_seconds"] == 2
    assert result.metrics["prefill_seconds"] == 0.5
    assert result.metrics["input_tokens"] == 30
    assert result.metrics["cached_tokens"] == "not_reported"
    assert result.metrics["first_token_seconds"] >= 0
    assert "thinking" not in repr(result)
    assert "secret" not in repr(result)
    payload = stub.calls[-1][2]
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["format"] == {"type": "object"}
    assert payload["options"]["num_predict"] == 120


def test_completion_length_is_reported_for_engine_to_reject():
    stub = OllamaStub()
    stub.stream[-1]["done_reason"] = "length"
    with stub.client() as client:
        assert client.chat("", "question").done_reason == "length"


@pytest.mark.parametrize(
    "values, expected",
    [
        ({"prompt_eval_cached_count": 12}, 12),
        ({"cache_read_input_tokens": 9}, 9),
        ({"prompt_eval_cached_count": 12, "cache_read_input_tokens": 9}, 12),
        ({"prompt_eval_cached_count": 0, "cache_read_input_tokens": 9}, 0),
    ],
)
def test_cache_counts_are_only_taken_from_reported_fields(values, expected):
    stub = OllamaStub()
    stub.stream[-1].update(values)
    with stub.client() as client:
        assert client.chat("", "question").metrics["cached_tokens"] == expected


def test_verify_cache_is_invalidated_when_digest_changes():
    stub = OllamaStub()
    with stub.client() as client:
        assert client.model_digest == DIGEST
        assert client.model_digest == DIGEST
        assert sum(path == "/api/show" for _, path, _ in stub.calls) == 1
        stub.digest = "c" * 64
        assert client.model_digest == "c" * 64
        assert sum(path == "/api/show" for _, path, _ in stub.calls) == 2


def test_digest_replaced_during_metadata_verification_never_receives_question():
    stub = OllamaStub()

    def replace_during_show(request):
        response = stub.handle(request)
        if request.url.path == "/api/show":
            stub.digest = "c" * 64
        return response

    with (
        InstrumentedOllamaClient(
            "http://localhost:11435",
            MODEL,
            [MODEL],
            transport=httpx.MockTransport(replace_during_show),
        ) as client,
        pytest.raises(LocalModelSecurityError),
    ):
        client.chat("private", "secret question")
    assert not any(path == "/api/chat" for _, path, _ in stub.calls)


@pytest.mark.parametrize(
    "metadata",
    [
        {**local_metadata(), "remote_host": "https://external.test"},
        {**local_metadata(), "remote_model": "some-model"},
        {"details": {"format": "gguf"}},
        {},
    ],
)
def test_unverified_or_remote_model_never_receives_prompt(metadata):
    stub = OllamaStub()
    stub.metadata = metadata
    with stub.client() as client, pytest.raises(LocalModelSecurityError):
        client.chat("private", "secret question")
    assert not any(path == "/api/chat" for _, path, _ in stub.calls)


def test_missing_digest_never_receives_prompt():
    stub = OllamaStub()
    stub.digest = None
    with stub.client() as client, pytest.raises(LocalModelSecurityError):
        client.chat("", "secret question")
    assert not any(path in {"/api/show", "/api/chat"} for _, path, _ in stub.calls)


def test_redirect_is_rejected_and_never_followed():
    calls = []

    def redirect(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://external.test"})

    with InstrumentedOllamaClient(
        "http://localhost:11435",
        MODEL,
        [MODEL],
        transport=httpx.MockTransport(redirect),
    ) as client:
        with pytest.raises(LocalModelSecurityError):
            client.chat("", "question")
    assert calls == ["http://127.0.0.1:11435/api/tags"]


def test_proxy_environment_is_not_used(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://external.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://external.invalid:9999")
    with OllamaStub().client() as client:
        assert not client._http.trust_env
        assert client._http.follow_redirects is False
        assert client.chat("", "question").content


@pytest.mark.parametrize(
    "stream",
    [
        [chunk("unfinished")],
        [dict(chunk("wrong", done=True), model="different:latest")],
        [{"error": "private server response"}],
    ],
)
def test_invalid_stream_is_never_returned_as_an_answer(stream):
    stub = OllamaStub()
    stub.stream = stream
    with stub.client() as client, pytest.raises(LocalModelError) as exc:
        client.chat("", "question")
    assert "private server response" not in str(exc.value)


def test_embedding_uses_same_local_verification_and_never_truncates():
    stub = OllamaStub()
    with SafeEmbeddingClient(
        "http://localhost:11435",
        "embeddinggemma",
        ["embeddinggemma"],
        transport=httpx.MockTransport(stub.handle),
    ) as client:
        assert client.embed_texts(["one", "two"]) == [[1.0, 0.0], [1.0, 0.0]]
        assert client.model_digest == "b" * 64
        assert client.last_metrics["model"] == "embeddinggemma"
        client.warmup()
    calls = [body for _, path, body in stub.calls if path == "/api/embed"]
    assert all(body["truncate"] is False for body in calls)


def test_warmup_and_health_send_no_user_questions():
    stub = OllamaStub()
    with stub.client() as client:
        assert client.warmup()["digest"] == DIGEST
    health = inspect_health(
        "http://localhost:11435",
        [MODEL, "embeddinggemma", "missing:1b"],
        transport=httpx.MockTransport(stub.handle),
    )
    assert health["status"] == "connected"
    assert health["models"][0]["local"] is True
    assert health["models"][-1]["installed"] is False
    assert not any(path == "/api/chat" for _, path, _ in stub.calls)


def test_per_call_deadline_is_applied_to_stream():
    stub = OllamaStub()
    timeouts = []

    def record(request):
        if request.url.path == "/api/chat":
            timeouts.append(request.extensions["timeout"]["read"])
        return stub.handle(request)

    with InstrumentedOllamaClient(
        "http://localhost:11435", MODEL, [MODEL], transport=httpx.MockTransport(record)
    ) as client:
        client.chat("", "question", timeout_seconds=2)
    assert 0 < timeouts[0] <= 2


@pytest.mark.parametrize("override, expected_budget", [(None, 45), (2, 2)])
def test_embedding_uses_one_deadline_across_verification_and_request(
    monkeypatch,
    override,
    expected_budget,
):
    stub = OllamaStub()
    now = [10.0]
    monkeypatch.setattr("src.lab.client.time.perf_counter", lambda: now[0])
    timeouts = []

    def elapsed_request(request):
        timeouts.append(request.extensions["timeout"]["read"])
        now[0] += 0.2
        return stub.handle(request)

    with SafeEmbeddingClient(
        "http://localhost:11435",
        "embeddinggemma",
        ["embeddinggemma"],
        timeout_seconds=45,
        transport=httpx.MockTransport(elapsed_request),
    ) as client:
        assert client.embed_texts(["question"], timeout_seconds=override) == [
            [1.0, 0.0]
        ]
        assert client.last_metrics["wall_seconds"] == pytest.approx(0.8)
    assert timeouts[0] == expected_budget
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[-1] == pytest.approx(expected_budget - 0.6)


@pytest.mark.parametrize("slow_stage", ["/api/tags", "/api/show", "/api/embed"])
def test_embedding_rejects_response_after_total_deadline(monkeypatch, slow_stage):
    stub = OllamaStub()
    now = [10.0]
    monkeypatch.setattr("src.lab.client.time.perf_counter", lambda: now[0])

    def slow_request(request):
        if request.url.path == slow_stage:
            now[0] += 3
        return stub.handle(request)

    with (
        SafeEmbeddingClient(
            "http://localhost:11435",
            "embeddinggemma",
            ["embeddinggemma"],
            transport=httpx.MockTransport(slow_request),
        ) as client,
        pytest.raises(LocalModelError, match="待ち時間"),
    ):
        client.embed_texts(["question"], timeout_seconds=2)
    assert client.last_metrics == {}
    if slow_stage != "/api/embed":
        assert not any(path == "/api/embed" for _, path, _ in stub.calls)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), True])
def test_embedding_rejects_invalid_deadline_before_network(timeout):
    stub = OllamaStub()
    with (
        SafeEmbeddingClient(
            "http://localhost:11435",
            "embeddinggemma",
            ["embeddinggemma"],
            transport=httpx.MockTransport(stub.handle),
        ) as client,
        pytest.raises(ValueError),
    ):
        client.embed_texts(["question"], timeout_seconds=timeout)
    assert not stub.calls
