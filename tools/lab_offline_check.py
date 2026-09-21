"""Exercise the local application while denying non-loopback Python sockets.

This is not an OS firewall, daemon egress monitor, or physical air-gap proof.
Only fixed, hash-pinned fictional sample files enter an isolated diagnostic index.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import socket
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_HASHES = {
    "black_hole.txt": "63708e393508507904eece2005185f4bb28c4e425bc53f12c0763d8b4c4f8402",
    "moon_phase.txt": "5a9789ac5e548bc9f6717bb4a5bc986140e6c12ebb08e776a5a06ade9554deac",
    "museum_faq.txt": "bc70ffaa5b90f80b75398c1b518f4ddbb8408813a27d2988bfeaa9cd4af168a2",
    "planetarium_guide.txt": "6ceb4bb9764b0e52f08ca08181530fdca22bdac13641abc3b9048bb879e1d7a1",
    "solar_system.txt": "714d3e075e863bca8fd36b5713678173c3dcdc93d74b736f6b4db2b29ee2de24",
}
KNOWN_QUESTION = "月はなぜ明るく見えるのですか？"
OUTSIDE_QUESTION = "私の銀行口座の残高はいくらですか？"
KNOWN_QUOTE = "月は自分で光っているのではなく、太陽の光を反射して明るく見えます。"


class OfflineEgressBlocked(OSError):
    """Rejected before calling the operating system's DNS/connect function."""


class LoopbackSocketGuard:
    """Process-wide temporary guard for this single-process diagnostic CLI."""

    def __init__(self, port: int = 11435):
        self.port = port
        self.events: list[dict] = []
        self._originals: list[tuple[object, str, object]] = []

    def _local_address(self, host, *, operation: str) -> str:
        if isinstance(host, bytes):
            try:
                host = host.decode("ascii")
            except UnicodeError:
                host = None
        if host == "localhost":
            return "127.0.0.1"
        try:
            address = ipaddress.ip_address(host)
            if address.is_loopback and "%" not in str(address):
                return str(address)
        except (ValueError, TypeError):
            pass
        # Do not retain arbitrary input hostnames, which could contain sensitive data.
        self.events.append(
            {
                "operation": operation,
                "allowed": False,
                "endpoint": "non_loopback_or_dns",
            }
        )
        raise OfflineEgressBlocked(
            "Non-loopback network access is disabled in this diagnostic process."
        )

    def _endpoint(self, address, *, operation: str) -> tuple[str, int]:
        if not isinstance(address, tuple) or len(address) < 2:
            self.events.append(
                {
                    "operation": operation,
                    "allowed": False,
                    "endpoint": "unsupported_socket_address",
                }
            )
            raise OfflineEgressBlocked(
                "Only the dedicated loopback endpoint is allowed."
            )
        host = self._local_address(address[0], operation=operation)
        port = address[1]
        if isinstance(port, str) and port.isdecimal():
            port = int(port)
        if type(port) is not int or port != self.port:
            self.events.append(
                {
                    "operation": operation,
                    "allowed": False,
                    "endpoint": "unapproved_loopback_port",
                }
            )
            raise OfflineEgressBlocked("Only the dedicated Ollama port is allowed.")
        return host, port

    def _install(self, target, name, replacement):
        self._originals.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    def __enter__(self):
        if self._originals:
            raise RuntimeError("Socket guard cannot be entered twice.")
        original_getaddrinfo = socket.getaddrinfo
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_create_connection = socket.create_connection
        original_sendto = socket.socket.sendto

        def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
            local, number = self._endpoint((host, port), operation="resolve")
            # Numeric resolution does not ask a DNS server, including for localhost.
            return original_getaddrinfo(
                local, number, family, type, proto, flags | socket.AI_NUMERICHOST
            )

        def connect(sock, address):
            local, port = self._endpoint(address, operation="connect")
            self.events.append(
                {"operation": "connect", "allowed": True, "endpoint": f"{local}:{port}"}
            )
            return original_connect(sock, (local, port, *address[2:]))

        def connect_ex(sock, address):
            local, port = self._endpoint(address, operation="connect_ex")
            self.events.append(
                {
                    "operation": "connect_ex",
                    "allowed": True,
                    "endpoint": f"{local}:{port}",
                }
            )
            return original_connect_ex(sock, (local, port, *address[2:]))

        def create_connection(address, *args, **kwargs):
            self._endpoint(address, operation="create_connection")
            return original_create_connection(address, *args, **kwargs)

        def sendto(sock, *args):
            address = args[-1] if args else None
            local, port = self._endpoint(address, operation="sendto")
            return original_sendto(sock, *args[:-1], (local, port, *address[2:]))

        def gethostbyname(host):
            local = self._local_address(host, operation="resolve")
            if ipaddress.ip_address(local).version != 4:
                raise OfflineEgressBlocked("IPv4 address required.")
            return local

        def gethostbyname_ex(host):
            local = gethostbyname(host)
            return "localhost", [], [local]

        def gethostbyaddr(host):
            local = self._local_address(host, operation="reverse_resolve")
            return "localhost", [], [local]

        def getnameinfo(address, flags):
            local, port = self._endpoint(address, operation="reverse_resolve")
            return local, str(port)

        self._install(socket, "getaddrinfo", getaddrinfo)
        self._install(socket, "create_connection", create_connection)
        self._install(socket, "gethostbyname", gethostbyname)
        self._install(socket, "gethostbyname_ex", gethostbyname_ex)
        self._install(socket, "gethostbyaddr", gethostbyaddr)
        self._install(socket, "getnameinfo", getnameinfo)
        self._install(socket.socket, "connect", connect)
        self._install(socket.socket, "connect_ex", connect_ex)
        self._install(socket.socket, "sendto", sendto)
        return self

    def __exit__(self, *_args):
        for target, name, original in reversed(self._originals):
            setattr(target, name, original)
        self._originals.clear()


def _fixed_samples(project_root: Path) -> dict[str, bytes]:
    directory = (project_root / "sample_docs").resolve(strict=True)
    if not directory.is_relative_to(project_root):
        raise ValueError("Samples must stay inside the comparison project.")
    samples = {}
    for name, expected in SAMPLE_HASHES.items():
        path = (directory / name).resolve(strict=True)
        if path.parent != directory or not path.is_file():
            raise ValueError("Bundled sample path is not valid.")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError(
                "Bundled samples have changed; review their fixed hashes first."
            )
        samples[name] = body
    return samples


def _probe(probes, name, passed, *, status=None, elapsed=None):
    record = {"name": name, "passed": bool(passed)}
    if status is not None:
        record["status"] = status
    if elapsed is not None:
        record["seconds"] = round(elapsed, 6)
    probes.append(record)


def run_diagnostics(
    project_root=PROJECT_ROOT, *, config_path=None, model=None, bundle_factory=None
):
    """Write only a fresh diagnostic directory; never open the production index."""
    root = Path(project_root).resolve(strict=True)
    samples = _fixed_samples(root)
    output_parent = (root / "outputs" / "offline-check").resolve()
    if not output_parent.is_relative_to(root):
        raise ValueError("Diagnostic output must stay inside the comparison project.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]
    output = output_parent / run_id
    output.mkdir(parents=True, exist_ok=False)
    probes: list[dict] = []
    report = {
        "schema_version": 1,
        "scope": "application_python_process_socket_guard",
        "physical_airgap_verified": False,
        "ollama_process_egress_monitored": False,
        "browser_and_os_egress_monitored": False,
        "native_extensions_and_subprocesses_monitored": False,
        "explanation_semantic_support_verified": False,
        "sample_count": len(samples),
        "probes": probes,
    }
    bundle = None
    start = time.perf_counter()
    with LoopbackSocketGuard() as guard:
        try:
            try:
                socket.getaddrinfo("blocked.invalid", 443)
                _probe(probes, "external_dns_denied", False)
            except OfflineEgressBlocked:
                _probe(probes, "external_dns_denied", True)
            try:
                with socket.socket() as probe_socket:
                    probe_socket.connect(("192.0.2.1", 443))
                _probe(probes, "external_connect_denied", False)
            except OfflineEgressBlocked:
                _probe(probes, "external_connect_denied", True)
            expected_blocks = sum(not item["allowed"] for item in guard.events)
            from src.lab.contracts import Claim
            from src.lab.settings import load_lab_settings

            path = (
                Path(config_path).resolve(strict=True)
                if config_path
                else root / "lab_config.yaml"
            )
            settings = load_lab_settings(path)
            if settings.ollama.base_url != "http://127.0.0.1:11435":
                raise ValueError(
                    "Diagnostic requires the dedicated loopback Ollama endpoint."
                )
            if bundle_factory is None:
                from src.lab.factory import build_lab

                bundle_factory = build_lab
            bundle = bundle_factory(
                path, data_dir=output / "index", generation_model=model
            )
            report["model"] = bundle.client.model_name
            report["model_digest"] = bundle.client.model_digest
            report["embedding_model"] = bundle.embedding.model_name
            report["embedding_digest"] = bundle.embedding.model_digest
            registration_start = time.perf_counter()
            for name, body in samples.items():
                bundle.index.register_bytes(name, body, approved=True)
            _probe(
                probes,
                "fictional_samples_registered",
                True,
                elapsed=time.perf_counter() - registration_start,
            )

            answer_start = time.perf_counter()
            answer = bundle.engine.answer(KNOWN_QUESTION, mode="quoted")
            supported = (
                answer.status == "answered"
                and bool(answer.claims)
                and "太陽" in answer.text
                and "反射" in answer.text
                and all(
                    claim.text == claim.quote
                    and any(
                        evidence.evidence_id == claim.evidence_id
                        and claim.quote in evidence.text
                        for evidence in answer.evidence
                    )
                    for claim in answer.claims
                )
            )
            _probe(
                probes,
                "local_grounded_answer",
                supported,
                status=answer.status,
                elapsed=time.perf_counter() - answer_start,
            )
            report["answer_generation_calls"] = len(answer.calls)

            refusal_start = time.perf_counter()
            refusal = bundle.engine.answer(OUTSIDE_QUESTION, mode="quoted")
            _probe(
                probes,
                "outside_question_refused",
                refusal.status == "refused" and not refusal.claims,
                status=refusal.status,
                elapsed=time.perf_counter() - refusal_start,
            )

            # Explanation text need not equal a quote. This checks only citation
            # provenance and bounded generation/review, not semantic entailment.
            explanation_start = time.perf_counter()
            explanation = bundle.engine.answer(KNOWN_QUESTION, mode="explain")
            citations_present = (
                explanation.status == "answered"
                and bool(explanation.claims)
                and explanation.route == "explain"
                and [call.get("stage") for call in explanation.calls]
                == ["explanation", "verification"]
                and all(
                    bool(claim.references)
                    and all(
                        reference.quote
                        and any(
                            evidence.evidence_id == reference.evidence_id
                            and reference.quote in evidence.text
                            for evidence in explanation.evidence
                        )
                        for reference in claim.references
                    )
                    for claim in explanation.claims
                )
            )
            _probe(
                probes,
                "local_explanation_citations",
                citations_present,
                status=explanation.status,
                elapsed=time.perf_counter() - explanation_start,
            )
            report["explanation_model_calls"] = len(explanation.calls)
            report["explanation_generation_calls"] = sum(
                call.get("stage") == "explanation" for call in explanation.calls
            )
            report["explanation_verification_calls"] = sum(
                call.get("stage") == "verification" for call in explanation.calls
            )
            explanation_refusal_start = time.perf_counter()
            explanation_refusal = bundle.engine.answer(OUTSIDE_QUESTION, mode="explain")
            _probe(
                probes,
                "explanation_outside_question_refused",
                explanation_refusal.status == "refused"
                and not explanation_refusal.claims,
                status=explanation_refusal.status,
                elapsed=time.perf_counter() - explanation_refusal_start,
            )

            passage = next(
                item
                for item in bundle.index.export_passages()
                if item.source_name == "moon_phase.txt" and KNOWN_QUOTE in item.text
            )
            # This is a synthetic approval confined to this fictional diagnostic index.
            claim = Claim(KNOWN_QUOTE, passage.evidence_id, KNOWN_QUOTE)
            bundle.faq.approve(
                KNOWN_QUESTION,
                (claim,),
                (passage,),
                expires=(date.today() + timedelta(days=1)).isoformat(),
            )
            faq_start = time.perf_counter()
            reused = bundle.engine.answer(KNOWN_QUESTION, mode="quoted")
            _probe(
                probes,
                "synthetic_faq_reused_without_generation",
                reused.status == "answered"
                and reused.route == "approved_faq"
                and not reused.calls,
                status=reused.status,
                elapsed=time.perf_counter() - faq_start,
            )
            updated = (
                samples["moon_phase.txt"]
                + "\n【架空のオフライン検査用更新】資料の確認版を更新しました。\n".encode()
            )
            bundle.index.register_bytes("moon_phase.txt", updated, approved=True)
            stale = bundle.faq.lookup(KNOWN_QUESTION, bundle.index.export_passages())
            _probe(probes, "updated_source_invalidates_faq", stale is None)
            _probe(
                probes,
                "no_unexpected_application_egress",
                sum(not item["allowed"] for item in guard.events) == expected_blocks,
            )
        except Exception as exc:
            # Never serialize exception bodies, raw prompts, or model responses.
            _probe(probes, "diagnostic_completed", False, status=type(exc).__name__)
        finally:
            if bundle is not None:
                try:
                    bundle.close()
                except Exception as exc:
                    _probe(probes, "resources_closed", False, status=type(exc).__name__)
    report["connections"] = guard.events
    report["total_seconds"] = round(time.perf_counter() - start, 6)
    report["passed"] = bool(probes) and all(item["passed"] for item in probes)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="同梱の架空資料だけで、Pythonプロセスの外部通信拒否とローカル回答・FAQを検査します。OSやOllamaの通信遮断を証明するものではありません。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "lab_config.yaml",
        help="比較版の設定ファイル",
    )
    parser.add_argument("--model", help="設定で許可されたインストール済み回答モデル")
    args = parser.parse_args(argv)
    try:
        report, output = run_diagnostics(config_path=args.config, model=args.model)
    except Exception as exc:
        print(
            json.dumps(
                {"passed": False, "error_type": type(exc).__name__}, ensure_ascii=False
            )
        )
        return 1
    print(
        json.dumps(
            {"report_path": str(output / "report.json"), **report},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
