"""Socket guard and diagnostics operate on fakes without local model calls."""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.lab.contracts import Claim, Evidence, LabAnswer, SourceCitation
from src.lab.faq import ApprovedFAQ
from tools.lab_offline_check import (
    KNOWN_QUESTION,
    KNOWN_QUOTE,
    OUTSIDE_QUESTION,
    PROJECT_ROOT,
    SAMPLE_HASHES,
    LoopbackSocketGuard,
    OfflineEgressBlocked,
    run_diagnostics,
)

EXPLANATION = "月は太陽の光を反射するので、私たちには明るく見えます。"


def test_external_dns_and_connect_never_reach_original_socket(monkeypatch):
    reached = []
    original_getaddrinfo = socket.getaddrinfo

    def unexpected(*args, **kwargs):
        reached.append(True)
        raise AssertionError("Operating system network function must not run")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected)
    monkeypatch.setattr(socket.socket, "connect", unexpected)
    monkeypatch.setattr(socket.socket, "connect_ex", unexpected)
    monkeypatch.setattr(socket.socket, "sendto", unexpected)
    with LoopbackSocketGuard() as guard:
        with pytest.raises(OfflineEgressBlocked):
            socket.getaddrinfo("private-data.invalid", 443)
        with socket.socket() as sock:
            for method in (sock.connect, sock.connect_ex):
                with pytest.raises(OfflineEgressBlocked):
                    method(("192.0.2.1", 443))
            with pytest.raises(OfflineEgressBlocked):
                sock.sendto(b"test", ("192.0.2.1", 53))
        with pytest.raises(OfflineEgressBlocked):
            socket.create_connection(("example.invalid", 80))
        assert "private-data" not in json.dumps(guard.events)
    assert socket.getaddrinfo is unexpected
    assert not reached
    assert original_getaddrinfo is not socket.getaddrinfo


def test_allowed_loopback_is_numeric_and_other_local_ports_are_blocked(monkeypatch):
    received = []

    def local_connect(sock, address):
        received.append(address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    with LoopbackSocketGuard() as guard:
        with socket.socket() as sock:
            sock.connect(("localhost", 11435))
            with pytest.raises(OfflineEgressBlocked):
                sock.connect(("127.0.0.1", 11434))
        assert socket.gethostbyname("localhost") == "127.0.0.1"
        assert socket.gethostbyaddr("127.0.0.1")[0] == "localhost"
        assert socket.getnameinfo(("127.0.0.1", 11435), 0) == ("127.0.0.1", "11435")
        assert any(event["allowed"] for event in guard.events)
    assert received == [("127.0.0.1", 11435)]


def test_numeric_resolution_forces_no_dns(monkeypatch):
    arguments = []

    def resolver(*args):
        arguments.append(args)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    with LoopbackSocketGuard():
        socket.getaddrinfo("localhost", 11435)
    assert arguments[0][0] == "127.0.0.1"
    assert arguments[0][-1] & socket.AI_NUMERICHOST


def test_guard_restores_sockets_after_failure():
    before = (socket.getaddrinfo, socket.socket.connect, socket.gethostbyname)
    with pytest.raises(RuntimeError), LoopbackSocketGuard():
        raise RuntimeError("injected failure")
    assert before == (socket.getaddrinfo, socket.socket.connect, socket.gethostbyname)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "sample_docs").mkdir()
    for name in SAMPLE_HASHES:
        (tmp_path / "sample_docs" / name).write_bytes(
            (PROJECT_ROOT / "sample_docs" / name).read_bytes()
        )
    (tmp_path / "lab_config.yaml").write_text("{}", encoding="utf-8")
    return tmp_path


class FakeIndex:
    def __init__(self):
        self.passages = []

    def register_bytes(self, name, body, **kwargs):
        digest = hashlib.sha256(body).hexdigest()
        passage = Evidence(digest, body.decode(), name, None, digest)
        self.passages = [item for item in self.passages if item.source_name != name] + [
            passage
        ]

    def export_passages(self):
        return tuple(self.passages)


class FakeBundle:
    def __init__(self, root):
        self.index = FakeIndex()
        self.faq = ApprovedFAQ(root / "faq.sqlite3")
        self.client = SimpleNamespace(model_name="fake:local", model_digest="a" * 64)
        self.embedding = SimpleNamespace(
            model_name="fake:embedding", model_digest="b" * 64
        )
        self.engine = self
        self.closed = False
        self.answer_modes = []

    def answer(self, question, **kwargs):
        self.answer_modes.append(kwargs["mode"])
        reused = self.faq.lookup(question, self.index.export_passages())
        if reused:
            return reused
        if question == OUTSIDE_QUESTION:
            return LabAnswer("refused", "PRIVATE_ANSWER_MUST_NOT_APPEAR")
        source = next(item for item in self.index.passages if KNOWN_QUOTE in item.text)
        if kwargs["mode"] == "explain":
            return LabAnswer(
                "answered",
                "",
                (
                    Claim(
                        EXPLANATION,
                        source.evidence_id,
                        KNOWN_QUOTE,
                        (SourceCitation(source.evidence_id, KNOWN_QUOTE),),
                    ),
                ),
                (source,),
                route="explain",
                calls=(
                    {"model": "fake", "stage": "explanation"},
                    {"model": "fake", "stage": "verification"},
                ),
            )
        return LabAnswer(
            "answered",
            "",
            (Claim(KNOWN_QUOTE, source.evidence_id, KNOWN_QUOTE),),
            (source,),
            calls=({"model": "fake"},),
        )

    def close(self):
        self.closed = True


def test_full_diagnostic_with_fakes_uses_unique_isolated_output_and_no_text(project):
    bundles = []
    roots = []

    def factory(config, *, data_dir, generation_model):
        roots.append(data_dir)
        bundle = FakeBundle(data_dir)
        bundles.append(bundle)
        return bundle

    report, output = run_diagnostics(project, bundle_factory=factory)
    second, output2 = run_diagnostics(project, bundle_factory=factory)
    assert report["passed"] and second["passed"]
    assert output != output2
    assert output.parent == project / "outputs/offline-check"
    assert roots[0] == output / "index"
    assert not (project / "data").exists()
    assert all(bundle.closed for bundle in bundles)
    assert all(
        bundle.answer_modes == ["quoted", "quoted", "explain", "explain", "quoted"]
        for bundle in bundles
    )
    assert report["physical_airgap_verified"] is False
    assert report["explanation_semantic_support_verified"] is False
    assert report["explanation_generation_calls"] == 1
    assert report["explanation_verification_calls"] == 1
    assert report["explanation_model_calls"] == 2
    serialized = (output / "report.json").read_text(encoding="utf-8")
    for private in (
        KNOWN_QUESTION,
        OUTSIDE_QUESTION,
        KNOWN_QUOTE,
        EXPLANATION,
        "PRIVATE_ANSWER_MUST_NOT_APPEAR",
    ):
        assert private not in serialized
    assert all(item["passed"] for item in report["probes"])
    assert {item["name"] for item in report["probes"]} >= {
        "local_grounded_answer",
        "outside_question_refused",
        "local_explanation_citations",
        "explanation_outside_question_refused",
        "synthetic_faq_reused_without_generation",
    }


@pytest.mark.parametrize("bad_reference", [None, "missing_quote", "missing_source"])
def test_explanation_probe_rejects_missing_or_fabricated_citations(
    project, bad_reference
):
    class InvalidExplanationBundle(FakeBundle):
        def answer(self, question, **kwargs):
            answer = super().answer(question, **kwargs)
            if kwargs["mode"] != "explain" or question != KNOWN_QUESTION:
                return answer
            references = (
                ()
                if bad_reference is None
                else (
                    SourceCitation(
                        "unknown"
                        if bad_reference == "missing_source"
                        else answer.evidence[0].evidence_id,
                        "NOT_IN_SOURCE"
                        if bad_reference == "missing_quote"
                        else KNOWN_QUOTE,
                    ),
                )
            )
            return LabAnswer(
                "answered",
                "",
                (Claim(EXPLANATION, "", "", references),),
                answer.evidence,
                route="explain",
                calls=answer.calls,
            )

    report, _ = run_diagnostics(
        project,
        bundle_factory=lambda config, *, data_dir, generation_model: (
            InvalidExplanationBundle(data_dir)
        ),
    )
    assert report["passed"] is False
    probe = next(
        item
        for item in report["probes"]
        if item["name"] == "local_explanation_citations"
    )
    assert probe["passed"] is False
    assert probe["status"] == "answered"


@pytest.mark.parametrize(
    "stages",
    [
        ("explanation",),
        ("verification",),
        ("explanation", "explanation"),
        ("verification", "explanation"),
        ("explanation", "verification", "verification"),
    ],
)
def test_explanation_probe_requires_one_generation_then_one_review(project, stages):
    class WrongCallCountBundle(FakeBundle):
        def answer(self, question, **kwargs):
            answer = super().answer(question, **kwargs)
            if kwargs["mode"] == "explain" and question == KNOWN_QUESTION:
                return replace(
                    answer,
                    calls=tuple({"model": "fake", "stage": stage} for stage in stages),
                )
            return answer

    report, _ = run_diagnostics(
        project,
        bundle_factory=lambda config, *, data_dir, generation_model: (
            WrongCallCountBundle(data_dir)
        ),
    )
    assert report["passed"] is False
    probe = next(
        item
        for item in report["probes"]
        if item["name"] == "local_explanation_citations"
    )
    assert probe["passed"] is False
    assert report["explanation_model_calls"] == len(stages)
    assert report["explanation_generation_calls"] == stages.count("explanation")
    assert report["explanation_verification_calls"] == stages.count("verification")


def test_modified_sample_is_rejected_before_any_bundle_or_network(project):
    (project / "sample_docs/moon_phase.txt").write_text("PRIVATE_DOC", encoding="utf-8")
    with pytest.raises(ValueError, match="samples have changed"):
        run_diagnostics(
            project,
            bundle_factory=lambda *args, **kwargs: pytest.fail("must not build"),
        )
    assert not (project / "outputs").exists()


def test_application_egress_failure_is_redacted_and_fails_report(project):
    def external_factory(*args, **kwargs):
        socket.getaddrinfo("secret-from-document.invalid", 443)

    report, output = run_diagnostics(project, bundle_factory=external_factory)
    assert report["passed"] is False
    assert report["probes"][-1]["status"] == "OfflineEgressBlocked"
    assert "secret-from-document" not in (output / "report.json").read_text()


def test_unverified_daemon_port_is_rejected_before_bundle(project):
    (project / "lab_config.yaml").write_text(
        "ollama:\n  base_url: http://127.0.0.1:11434", encoding="utf-8"
    )
    report, _ = run_diagnostics(
        project, bundle_factory=lambda *args, **kwargs: pytest.fail("must not build")
    )
    assert not report["passed"]
    assert report["probes"][-1]["status"] == "ValueError"
