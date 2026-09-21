"""Saved synthetic draft replay with fake clients, no model or network calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.lab.contracts import GenerationResult
from tools import lab_verify_replay as replay

QUESTION = "PRIVATE_QUESTION"
TEXT = "PRIVATE_DRAFT"
QUOTES = ("PRIVATE_QUOTE_A", "PRIVATE_QUOTE_B")


def row(case_id="case-1"):
    return {
        "case_id": case_id,
        "mode": "explain",
        "status": "answered",
        "question": QUESTION,
        "claims": [
            {
                "text": TEXT,
                "evidence_id": "first",
                "quote": QUOTES[0],
                "citations": [
                    {"evidence_id": "first", "quote": QUOTES[0]},
                    {"evidence_id": "second", "quote": QUOTES[1]},
                ],
            }
        ],
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "lab_config.yaml").write_text("{}", encoding="utf-8")
    results = tmp_path / "outputs/lab-comparisons/synthetic/report/results.jsonl"
    write_rows(results, [row(), row("case-2")])
    return tmp_path, results


class FakeFactory:
    def __init__(self, replies=None):
        self.replies = iter(
            replies
            or [
                GenerationResult(
                    '{"verdict":"SUPPORTED"}',
                    "stop",
                    {
                        "model": "qwen3:1.7b",
                        "digest": "a" * 64,
                        "wall_seconds": 0.5,
                        "output_tokens": 9,
                        "cached_tokens": "not_reported",
                        "thinking": "PRIVATE_THINKING",
                        "raw_payload": "PRIVATE_RAW",
                    },
                )
            ]
        )
        self.constructed = []
        self.calls = []
        self.closed = False

    def __call__(self, *args, **kwargs):
        self.constructed.append((args, kwargs))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def chat(self, system, user, **kwargs):
        self.calls.append((system, json.loads(user), kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(project, factory, **kwargs):
    root, results = project
    return replay.run_replay(
        project_root=root,
        results_path=results,
        case_ids=kwargs.pop("case_ids", ["case-1"]),
        run_id=kwargs.pop("run_id", "replay-1"),
        client_factory=factory,
        **kwargs,
    )


def test_replay_preserves_all_quotes_and_records_bounded_review_not_accuracy(project):
    factory = FakeFactory()
    report, output = run(project, factory)
    assert report["completed"] is True
    assert report["unseen_evaluation"] is False
    assert report["semantic_support_guaranteed"] is False
    assert report["source_provenance_rechecked"] is False
    assert report["schema_version"] == 2
    assert report["replay_context"] == "recorded_quotes_only"
    assert report["full_source_context_available"] is False
    assert report["full_context_verifier_replayed"] is False
    assert "SUPPORTED verdict does not reproduce" in report["limitation"]
    assert (
        report["purpose"]
        == "exposed_synthetic_regression_replay_not_accuracy_evaluation"
    )
    assert (
        report["source_results_sha256"]
        == hashlib.sha256(project[1].read_bytes()).hexdigest()
    )
    assert all(len(value) == 64 for value in report["code_sha256"].values())
    assert set(report["code_sha256"]) == {
        "src/lab/client.py",
        "src/lab/explanation.py",
        "tools/lab_verify_replay.py",
    }
    assert len(factory.calls) == 1 and factory.closed
    _system, sent, options = factory.calls[0]
    assert sent["question"] == QUESTION
    assert len(sent["answer"]) == 1
    assert sent["answer"][0]["text"] == TEXT
    assert sent["answer"][0]["references"] == ["R1", "R2"]
    assert sent["citations"] == [
        {"id": "R1", "source_id": "first", "quote": QUOTES[0]},
        {"id": "R2", "source_id": "second", "quote": QUOTES[1]},
    ]
    assert sent["sources"] == []
    assert options["max_tokens"] == 32
    assert options["temperature"] == 0.0
    assert options["schema"]["required"] == ["verdict"]
    assert options["timeout_seconds"] == 90
    assert factory.constructed[0][0][0] == "http://127.0.0.1:11435"
    assert report["results"][0]["verdict"] == "SUPPORTED"
    assert report["results"][0]["replay_context"] == "recorded_quotes_only"
    assert report["results"][0]["status"] == "answered"
    assert report["results"][0]["metrics"]["digest"] == "a" * 64
    assert output.parent == project[0] / "outputs/verification-replays"
    assert not (project[0] / "data").exists()
    saved = (output / "report.json").read_text(encoding="utf-8")
    for private in (QUESTION, TEXT, *QUOTES, "PRIVATE_THINKING", "PRIVATE_RAW"):
        assert private not in saved
    assert json.loads(saved) == report


def test_quote_metadata_never_becomes_fabricated_full_passage_context(
    project, monkeypatch
):
    recorded = row()
    recorded["claims"][0]["citations"][0].update(
        source_name="unrelated_target.txt",
        content_hash="a" * 64,
        exact_quote_match=True,
    )
    # Unrecognized extra fields are not evidence of verified original passages.
    recorded["evidence"] = [{"text": "UNVERIFIED_SURROUNDING_CONTEXT"}]
    write_rows(project[1], [recorded])
    captured = []
    original_prompts = replay.verification_prompts

    def inspect_draft(question, draft):
        captured.append(draft)
        return original_prompts(question, draft)

    monkeypatch.setattr(replay, "verification_prompts", inspect_draft)
    factory = FakeFactory()
    report, _ = run(project, factory)
    assert captured[0].evidence == ()
    assert [reference.quote for reference in captured[0].claims[0].references] == list(
        QUOTES
    )
    assert "UNVERIFIED_SURROUNDING_CONTEXT" not in json.dumps(factory.calls)
    assert factory.calls[0][1]["sources"] == []
    assert report["results"][0]["verdict"] == "SUPPORTED"
    assert report["full_context_verifier_replayed"] is False
    assert report["full_source_context_available"] is False


def test_replay_selected_order_and_one_call_per_case_no_retries(project):
    factory = FakeFactory(
        [
            GenerationResult('{"verdict":"UNSUPPORTED"}', "stop"),
            GenerationResult('{"verdict":"SUPPORTED"}', "stop"),
        ]
    )
    report, _ = run(project, factory, case_ids=["case-2", "case-1"])
    assert report["completed"]
    assert [item["case_id"] for item in report["results"]] == ["case-2", "case-1"]
    assert [item["verdict"] for item in report["results"]] == [
        "UNSUPPORTED",
        "SUPPORTED",
    ]
    assert len(factory.calls) == 2
    assert report["retry_count"] == 0


@pytest.mark.parametrize(
    "verdict,status",
    [
        ("UNSUPPORTED", "needs_review"),
        ("INCOMPLETE", "needs_review"),
        ("AMBIGUOUS", "clarify"),
        ("CONFLICT", "refused"),
    ],
)
def test_negative_verdict_is_completed_diagnostic_not_correctness_failure(
    project, verdict, status
):
    factory = FakeFactory([GenerationResult(json.dumps({"verdict": verdict}), "stop")])
    report, _ = run(project, factory)
    assert report["completed"] is True
    assert report["results"][0]["verdict"] == verdict
    assert report["results"][0]["status"] == status


@pytest.mark.parametrize(
    "reply,issue",
    [
        (
            GenerationResult('{"verdict":"SUPPORTED"}', "length"),
            "verification_truncated",
        ),
        (GenerationResult("{", "stop"), "invalid_verification_json"),
        (
            GenerationResult('{"verdict":"UNSUPPORTED","verdict":"SUPPORTED"}', "stop"),
            "duplicate_json_key",
        ),
        (
            GenerationResult('{"verdict":"SUPPORTED","text":"PRIVATE_RAW"}', "stop"),
            "invalid_verification_schema",
        ),
        (RuntimeError("PRIVATE_EXCEPTION_BODY"), "RuntimeError"),
    ],
)
def test_failed_review_never_counts_as_supported_or_retries(project, reply, issue):
    factory = FakeFactory([reply])
    report, output = run(project, factory)
    assert report["completed"] is False
    assert report["results"][0]["verdict"] is None
    assert report["results"][0]["issues"] == [issue]
    assert len(factory.calls) == 1 and factory.closed
    saved = (output / "report.json").read_text(encoding="utf-8")
    assert "PRIVATE_EXCEPTION" not in saved and "PRIVATE_RAW" not in saved


def test_legacy_single_quote_fallback_is_replayed(project):
    recorded = row()
    del recorded["claims"][0]["citations"]
    write_rows(project[1], [recorded])
    factory = FakeFactory()
    report, _ = run(project, factory)
    assert report["completed"]
    sent = factory.calls[0][1]
    assert sent["answer"][0]["references"] == ["R1"]
    assert [citation["quote"] for citation in sent["citations"]] == [QUOTES[0]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "refused"),
        ("mode", "quoted"),
        ("question", ""),
        ("question", "Q" * 1501),
        ("claims", []),
        ("claims", [None]),
        ("claims", [{"text": "X" * 301, "evidence_id": "a", "quote": "q"}]),
        ("claims", [{"text": TEXT, "citations": []}]),
        ("claims", [{"text": TEXT, "citations": None}]),
        ("claims", [{"text": TEXT, "citations": [{"evidence_id": "a", "quote": ""}]}]),
    ],
)
def test_malformed_selected_records_stop_before_client_or_output(project, field, value):
    recorded = row()
    recorded[field] = value
    write_rows(project[1], [recorded])
    factory = FakeFactory()
    with pytest.raises(ValueError):
        run(project, factory)
    assert not factory.constructed
    assert not (project[0] / "outputs/verification-replays").exists()


@pytest.mark.parametrize(
    "case_ids", [[], ["missing"], ["case-1", "case-1"], ["../case"], "case-1"]
)
def test_invalid_or_missing_selection_stops_before_client(project, case_ids):
    factory = FakeFactory()
    with pytest.raises(ValueError):
        run(project, factory, case_ids=case_ids)
    assert not factory.constructed


def test_duplicate_selected_case_in_file_is_rejected_not_silently_chosen(project):
    write_rows(project[1], [row(), row()])
    factory = FakeFactory()
    with pytest.raises(ValueError, match="more than once"):
        run(project, factory)
    assert not factory.constructed


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "", ".", "a" * 97])
def test_invalid_run_id_is_rejected(project, run_id):
    factory = FakeFactory()
    with pytest.raises(ValueError):
        run(project, factory, run_id=run_id)
    assert not factory.constructed


def test_existing_output_is_never_overwritten_or_sent_again(project):
    first, output = run(project, FakeFactory())
    before = (output / "report.json").read_bytes()
    factory = FakeFactory()
    with pytest.raises(FileExistsError):
        run(project, factory)
    assert not factory.constructed
    assert (output / "report.json").read_bytes() == before
    assert first["completed"]


@pytest.mark.parametrize(
    "output_root", ["outputs/lab-comparisons", "data", "../outside"]
)
def test_output_must_stay_in_replay_area(project, output_root):
    factory = FakeFactory()
    with pytest.raises(ValueError, match="Replay output"):
        run(project, factory, output_root=output_root)
    assert not factory.constructed


def test_inputs_outside_comparison_area_are_rejected(project):
    root, _ = project
    other = root / "data/results.jsonl"
    write_rows(other, [row()])
    factory = FakeFactory()
    with pytest.raises(ValueError, match="Replay inputs"):
        run((root, other), factory)
    assert not factory.constructed


def test_large_input_rejected_before_read_or_network(project, monkeypatch):
    monkeypatch.setattr(replay, "MAX_RESULTS_BYTES", 4)
    factory = FakeFactory()
    with pytest.raises(ValueError, match="20 MiB"):
        run(project, factory)
    assert not factory.constructed


def test_cli_requires_explicit_input_selection_and_new_run_id():
    with pytest.raises(SystemExit) as exc:
        replay.main([])
    assert exc.value.code == 2


def test_cli_help_explains_quote_only_context_limitation(capsys):
    with pytest.raises(SystemExit) as exc:
        replay.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "引用文のみ" in help_text
    assert "復元しません" in help_text
    assert "全文文脈付き確認とは異なる" in help_text
    assert "SUPPORTED" in help_text


def test_cli_prints_only_failure_type_not_private_exception(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("PRIVATE_EXCEPTION_BODY")

    monkeypatch.setattr(replay, "run_replay", fail)
    code = replay.main(
        ["--results", "results.jsonl", "--case-id", "case-1", "--run-id", "new"]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out) == {
        "completed": False,
        "error_type": "RuntimeError",
    }


def test_cli_returns_success_for_completed_negative_verdict(monkeypatch, capsys):
    monkeypatch.setattr(
        replay, "run_replay", lambda **kwargs: ({"completed": True}, Path("output"))
    )
    code = replay.main(
        ["--results", "results.jsonl", "--case-id", "case-1", "--run-id", "new"]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["completed"] is True
    assert summary["replay_context"] == "recorded_quotes_only"
    assert summary["full_context_verifier_replayed"] is False
    assert summary["limitation"] == replay.REPLAY_LIMITATION
