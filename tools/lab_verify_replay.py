"""Replay exposed synthetic explanation cases against one bounded local reviewer.

This is a regression diagnostic, not an unseen accuracy benchmark. No retrieval,
answer generation, production index, retries, or source-document writes occur.
Reports identify the existing inputs by file hash and case ID without copying
questions, drafts, quotation text, raw reviewer responses, or model thoughts.

Only recorded quotes are replayed. The original passages, their headings, and
surrounding target context are not reconstructed. This does not reproduce the
current full-context verifier, even when the replay verdict is SUPPORTED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab import client as client_module  # noqa: E402
from src.lab import explanation as explanation_module  # noqa: E402
from src.lab.client import InstrumentedOllamaClient  # noqa: E402
from src.lab.contracts import Claim, LabAnswer, SourceCitation  # noqa: E402
from src.lab.explanation import (  # noqa: E402
    VERDICTS,
    apply_verification,
    verification_prompts,
    verification_schema,
)
from src.lab.grounding import unique_keys  # noqa: E402
from src.lab.settings import load_lab_settings  # noqa: E402

MAX_RESULTS_BYTES = 20 * 1024 * 1024
REPLAY_CONTEXT = "recorded_quotes_only"
REPLAY_LIMITATION = (
    "Only recorded quotes are supplied. Full original passages and their headings "
    "are unavailable and are not reconstructed. A SUPPORTED verdict does not "
    "reproduce or validate the full-context verifier and cannot establish target "
    "matching when it depends on omitted surrounding text."
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_NUMERIC_METRICS = {
    "wall_seconds",
    "total_seconds",
    "load_seconds",
    "prefill_seconds",
    "decode_seconds",
    "first_token_seconds",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
}


def _hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _text(value, limit: int, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Invalid recorded {label}.")
    return value


def _recorded_answer(row: dict) -> tuple[str, LabAnswer]:
    if row.get("status") != "answered" or row.get("mode") != "explain":
        raise ValueError("Selected records must contain answered explanation drafts.")
    question = _text(row.get("question"), 1500, "question")
    entries = row.get("claims")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 4:
        raise ValueError("Selected records must contain one to four claims.")
    claims = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid recorded claim.")
        text = _text(entry.get("text"), 300, "claim text")
        references = entry.get("citations")
        if "citations" not in entry:
            references = [
                {"evidence_id": entry.get("evidence_id"), "quote": entry.get("quote")}
            ]
        if not isinstance(references, list) or not 1 <= len(references) <= 4:
            raise ValueError("Each claim must have one to four recorded citations.")
        citations = []
        for reference in references:
            if not isinstance(reference, dict):
                raise ValueError("Invalid recorded citation.")
            citations.append(
                SourceCitation(
                    _text(reference.get("evidence_id"), 200, "evidence ID"),
                    _text(reference.get("quote"), 500, "quote"),
                )
            )
        claims.append(
            Claim(text, citations[0].evidence_id, citations[0].quote, tuple(citations))
        )
    if sum(len(claim.text) for claim in claims) > 800:
        raise ValueError("Recorded explanation exceeds its output contract.")
    # Citation metadata and exact_quote_match flags are not full source text.
    # Leave evidence empty; concatenating quotes would fabricate source context.
    return question, LabAnswer("answered", "", tuple(claims), route="explain")


def _load_records(path: Path, case_ids: tuple[str, ...]):
    if path.stat().st_size > MAX_RESULTS_BYTES:
        raise ValueError("Results file exceeds the 20 MiB replay input limit.")
    body = path.read_bytes()
    if len(body) > MAX_RESULTS_BYTES:
        raise ValueError("Results file exceeds the 20 MiB replay input limit.")
    selected = {}
    for line in body.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=unique_keys)
        if not isinstance(row, dict):
            raise ValueError("Results must contain JSON objects.")
        case_id = row.get("case_id")
        if isinstance(case_id, str) and case_id in case_ids:
            if case_id in selected:
                raise ValueError(
                    "A selected case ID occurs more than once; use an unambiguous run."
                )
            selected[case_id] = _recorded_answer(row)
    if set(selected) != set(case_ids):
        raise ValueError("A selected case ID is missing from the results file.")
    return [(case_id, *selected[case_id]) for case_id in case_ids], _hash(body)


def _safe_metrics(metrics, model):
    result = {}
    for key in _NUMERIC_METRICS:
        value = metrics.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[key] = value
        elif value == "not_reported":
            result[key] = value
    if metrics.get("model") == model:
        result["model"] = model
    digest = metrics.get("digest")
    if isinstance(digest, str) and re.fullmatch(r"(?:sha256:)?[a-fA-F0-9]{64}", digest):
        result["digest"] = digest
    return result


def run_replay(
    *,
    results_path,
    case_ids,
    run_id,
    output_root="outputs/verification-replays",
    config_path=None,
    project_root=PROJECT_ROOT,
    client_factory=None,
):
    """Validate all selected inputs before opening a model client or output run."""
    root = Path(project_root).resolve(strict=True)
    if not isinstance(run_id, str) or not _NAME.fullmatch(run_id):
        raise ValueError("run-id must be a simple directory name.")
    if isinstance(case_ids, (str, bytes)):
        raise ValueError("Select case IDs as a list.")
    case_ids = tuple(case_ids)
    if (
        not case_ids
        or any(
            not isinstance(item, str) or not _NAME.fullmatch(item) for item in case_ids
        )
        or len(case_ids) != len(set(case_ids))
    ):
        raise ValueError("Select at least one unique simple case ID.")
    allowed_input = (root / "outputs/lab-comparisons").resolve(strict=True)
    source = (root / Path(results_path)).resolve(strict=True)
    if (
        not allowed_input.is_relative_to(root)
        or not source.is_relative_to(allowed_input)
        or source.name != "results.jsonl"
        or not source.is_file()
    ):
        raise ValueError(
            "Replay inputs must be results.jsonl files inside outputs/lab-comparisons."
        )
    allowed_output = (root / "outputs/verification-replays").resolve()
    output_parent = (root / Path(output_root)).resolve()
    output = (output_parent / run_id).resolve()
    if (
        not allowed_output.is_relative_to(root)
        or not output_parent.is_relative_to(allowed_output)
        or output.parent != output_parent
    ):
        raise ValueError("Replay output must stay inside outputs/verification-replays.")
    if output.exists():
        raise FileExistsError("Replay run already exists; choose a new run-id.")
    records, source_hash = _load_records(source, case_ids)
    config = (root / Path(config_path or "lab_config.yaml")).resolve(strict=True)
    settings = load_lab_settings(config).ollama
    code_hashes = {
        "src/lab/explanation.py": _hash(Path(explanation_module.__file__).read_bytes()),
        "src/lab/client.py": _hash(Path(client_module.__file__).read_bytes()),
        "tools/lab_verify_replay.py": _hash(Path(__file__).read_bytes()),
    }
    report = {
        "schema_version": 2,
        "purpose": "exposed_synthetic_regression_replay_not_accuracy_evaluation",
        "replay_context": REPLAY_CONTEXT,
        "full_source_context_available": False,
        "full_context_verifier_replayed": False,
        "limitation": REPLAY_LIMITATION,
        "unseen_evaluation": False,
        "semantic_support_guaranteed": False,
        "source_provenance_rechecked": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_results": source.relative_to(root).as_posix(),
        "source_results_sha256": source_hash,
        "config_sha256": _hash(config.read_bytes()),
        "code_sha256": code_hashes,
        "model": settings.generation_model,
        "max_tokens": 32,
        "retry_count": 0,
        "case_ids": list(case_ids),
        "results": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    factory = client_factory or InstrumentedOllamaClient
    try:
        with factory(
            settings.base_url,
            settings.generation_model,
            settings.allowed_models,
            timeout_seconds=settings.timeout_seconds,
            context_length=settings.context_length,
        ) as client:
            for case_id, question, draft in records:
                record = {
                    "case_id": case_id,
                    "replay_context": REPLAY_CONTEXT,
                    "verdict": None,
                    "status": "error",
                    "issues": [],
                    "metrics": {},
                }
                try:
                    system, user = verification_prompts(question, draft)
                    reply = client.chat(
                        system,
                        user,
                        max_tokens=32,
                        schema=verification_schema(),
                        temperature=0.0,
                        timeout_seconds=settings.timeout_seconds,
                    )
                    record["metrics"] = _safe_metrics(
                        reply.metrics, settings.generation_model
                    )
                    record["done_reason"] = (
                        reply.done_reason
                        if reply.done_reason in {"stop", "length"}
                        else "unknown"
                    )
                    if reply.done_reason != "stop":
                        record.update(
                            status="needs_review", issues=["verification_truncated"]
                        )
                    else:
                        checked = apply_verification(reply.content, draft)
                        record.update(
                            status=checked.status, issues=list(checked.issues)
                        )
                        if checked.status == "answered":
                            record["verdict"] = "SUPPORTED"
                        elif len(checked.issues) == 1:
                            verdict = (
                                checked.issues[0].removeprefix("verification_").upper()
                            )
                            if verdict in VERDICTS:
                                record["verdict"] = verdict
                except Exception as exc:
                    record["issues"] = [type(exc).__name__]
                report["results"].append(record)
    except Exception as exc:
        report["setup_error_type"] = type(exc).__name__
    report["completed"] = (
        len(report["results"]) == len(case_ids)
        and all(record["verdict"] in VERDICTS for record in report["results"])
        and "setup_error_type" not in report
    )
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "保存済みの架空資料の説明案を、記録された引用文のみで1回ずつ再点検します。"
            "原文全体・見出し・前後の文脈は復元しません。"
            "現行の全文文脈付き確認とは異なる検査で、SUPPORTEDでも同方式の検証にはなりません。"
            "既知例の回帰検査であり、未見の正答率評価ではありません。"
        )
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="比較結果のresults.jsonl"
    )
    parser.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="再点検するcase ID（複数指定可）",
    )
    parser.add_argument(
        "--run-id", required=True, help="新しい検査名（既存結果の上書き不可）"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "lab_config.yaml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/verification-replays",
    )
    args = parser.parse_args(argv)
    try:
        report, output = run_replay(
            results_path=args.results,
            case_ids=args.case_id,
            run_id=args.run_id,
            config_path=args.config,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(json.dumps({"completed": False, "error_type": type(exc).__name__}))
        return 1
    print(
        json.dumps(
            {
                "completed": report["completed"],
                "report_path": str(output / "report.json"),
                "replay_context": REPLAY_CONTEXT,
                "full_context_verifier_replayed": False,
                "limitation": REPLAY_LIMITATION,
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
