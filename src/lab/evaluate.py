"""Synthetic, split-isolated comparison with explicit limits on automatic scoring.

Required-term checks measure spelling coverage, not factual correctness. Exact
quote matching establishes text provenance, not that a claim follows from it.
Every run therefore includes an unfilled human-review worksheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.lab.contracts import Claim, Evidence, LabAnswer

MODES = ("baseline", "quoted", "concise", "bounded", "explain")
AUDIENCES = ("general", "child", "staff")
DETAILS = ("short", "standard")
STATUSES = frozenset({"answered", "refused", "clarify", "needs_review", "error"})
CATEGORIES = frozenset(
    {
        "answerable",
        "missing_fact",
        "unrelated",
        "ambiguous",
        "contradiction",
        "outdated",
        "future",
        "document_injection",
        "numeric",
        "date",
        "comparison",
        "synthesis",
        "meaning_retention",
        "condition_retention",
        "wrong_target",
    }
)
LIMITATIONS = [
    "Synthetic self-authored museum data; no museum or subject-expert validation.",
    "Status agreement and required/forbidden terms are NOT semantic correctness.",
    "Exact quotation proves text presence, NOT scientific truth or entailment.",
    "Citation checks cover every reference, including secondary sources, but do not "
    "verify synthesis, meaning retention, audience suitability, or usefulness.",
    "Manual review is mandatory before claims about answer accuracy.",
    "This baseline reuses the lab corpus/SQLite dense index and the old two-call policy; "
    "it is not an exact rerun of the original Chroma/HNSW application.",
    "Model and prompt caches are not reset between queries. Latencies separate first "
    "queries, repeat exposure, and declared first-query preparation.",
    "Repeated attempts are not independent cases; denominators count attempts and "
    "unique case counts are reported separately.",
    "Holdout results are exploratory after inspection or repeated tuning; this suite "
    "does not establish generalization to real museum visitors.",
]


@dataclass(frozen=True)
class EvalCase:
    id: str
    split: str
    category: str
    question: str
    gold_sources: tuple[str, ...]
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    expected_status: tuple[str, ...]
    review_requirements: tuple[str, ...] = ()

    @property
    def answerable(self) -> bool:
        return "answered" in self.expected_status


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: str
    category: str
    question: str
    mode: str
    repeat: int
    temperature: str
    generation_exposure: str
    question_exposure: str
    expected_status: tuple[str, ...]
    status: str
    answer: str
    elapsed_seconds: float
    first_evidence_seconds: float | None
    timings: dict[str, float]
    calls: tuple[dict[str, float | int | str], ...]
    route: str
    issues: tuple[str, ...]
    claims: tuple[dict[str, Any], ...]
    evidence_sources: tuple[str, ...]
    gold_sources: tuple[str, ...]
    missing_terms: tuple[str, ...]
    forbidden_terms_found: tuple[str, ...]
    required_term_count: int
    exact_quote_count: int
    claim_equals_quote_count: int
    status_match: bool
    truncated: bool
    exception_type: str = ""
    audience: str = "general"
    detail: str = "standard"
    citation_count: int = 0
    exact_citation_count: int = 0
    review_requirements: tuple[str, ...] = ()
    presentation_options_applied: bool = False


class AnswerEngine(Protocol):
    def answer(
        self,
        question: str,
        mode: str,
        on_evidence=None,
        *,
        audience: str = "general",
        detail: str = "standard",
    ) -> LabAnswer: ...


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _strings(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of nonempty strings")
    if len(set(value)) != len(value) or (not allow_empty and not value):
        raise ValueError(f"{field} contains duplicates or is empty")
    return tuple(value)


def load_cases(path: Path | str, split: str | None = None) -> list[EvalCase]:
    """Validate the entire fixture before selecting a split; reject cross-split data."""
    path = Path(path)
    if split not in {None, "dev", "holdout"}:
        raise ValueError("split must be dev or holdout")
    corpus_root = (path.parent / "corpus").resolve()
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    fields = set(EvalCase.__dataclass_fields__)
    required_fields = fields - {"review_requirements"}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(raw, dict) or not required_fields <= set(raw) <= fields:
                raise ValueError("unexpected or missing case fields")
            raw.setdefault("review_requirements", [])
            for key in ("id", "split", "category", "question"):
                if not isinstance(raw[key], str) or not raw[key].strip():
                    raise ValueError(f"invalid {key}")
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,79}", raw["id"]):
                raise ValueError("invalid case id")
            if raw["split"] not in {"dev", "holdout"}:
                raise ValueError("unknown split")
            if raw["category"] not in CATEGORIES:
                raise ValueError("unknown category")
            for key in (
                "gold_sources",
                "required_terms",
                "forbidden_terms",
                "expected_status",
                "review_requirements",
            ):
                raw[key] = _strings(raw[key], key, allow_empty=key != "expected_status")
            if not set(raw["expected_status"]) <= STATUSES - {"error"}:
                raise ValueError("unknown or erroneous expected status")
            if "answered" in raw["expected_status"] and raw["expected_status"] != (
                "answered",
            ):
                raise ValueError(
                    "answerability must not mix answered with refusal statuses"
                )
            if "answered" in raw["expected_status"] and (
                not raw["gold_sources"] or not raw["required_terms"]
            ):
                raise ValueError(
                    "answerable cases need gold sources and required terms"
                )
            for source in raw["gold_sources"]:
                if "\\" in source or source.split("/")[0] != raw["split"]:
                    raise ValueError("gold source crosses split boundary")
                resolved = (corpus_root / source).resolve()
                if not resolved.is_relative_to(corpus_root / raw["split"]):
                    raise ValueError("gold source escapes its corpus")
                if not resolved.is_file():
                    raise ValueError(f"missing gold source: {source}")
            question_key = _normalized(raw["question"])
            if raw["id"] in seen_ids or question_key in seen_questions:
                raise ValueError("duplicate id or question across dataset")
            if set(map(_normalized, raw["required_terms"])) & set(
                map(_normalized, raw["forbidden_terms"])
            ):
                raise ValueError("a term is both required and forbidden")
            seen_ids.add(raw["id"])
            seen_questions.add(question_key)
            cases.append(EvalCase(**raw))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path.name}:{number}: {exc}") from exc
    if not cases:
        raise ValueError("dataset is empty")
    return [case for case in cases if split is None or case.split == split]


def load_corpus(dataset: Path | str, split: str) -> list[dict[str, Any]]:
    """Return only the selected split and explicit approval/date metadata."""
    root = Path(dataset).parent
    manifest = json.loads(
        (root / "corpus_manifest.json").read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if manifest.get("schema_version") != 1 or manifest.get("self_authored") is not True:
        raise ValueError("unrecognized corpus provenance")
    if split not in {"dev", "holdout"}:
        raise ValueError("unknown split")
    rows = manifest.get("documents")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty corpus manifest")
    seen: set[str] = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "source",
            "approved",
            "effective_date",
        }:
            raise ValueError("invalid corpus document entry")
        name = row["source"]
        if not isinstance(name, str) or name in seen or "\\" in name:
            raise ValueError("duplicate or invalid corpus path")
        parts = Path(name).parts
        if len(parts) != 2 or parts[0] not in {"dev", "holdout"} or ".." in parts:
            raise ValueError("corpus path escapes split")
        if not isinstance(row["approved"], bool):
            raise ValueError("approval metadata must be a boolean")
        if row["effective_date"] is not None:
            if not isinstance(row["effective_date"], str):
                raise ValueError("effective_date must be an ISO date or null")
            parsed_date = datetime.strptime(row["effective_date"], "%Y-%m-%d").date()
            if parsed_date.isoformat() != row["effective_date"]:
                raise ValueError("effective_date must be an ISO date")
        source = (root / "corpus" / name).resolve()
        if (
            not source.is_relative_to((root / "corpus" / parts[0]).resolve())
            or not source.is_file()
        ):
            raise ValueError("missing or escaped corpus file")
        seen.add(name)
        if parts[0] == split:
            result.append(
                {
                    **row,
                    "path": source,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
    actual = {
        file.relative_to(root / "corpus").as_posix()
        for file in (root / "corpus").glob("*/*")
        if file.is_file()
    }
    if actual != seen:
        raise ValueError("corpus files and manifest differ")
    return sorted(result, key=lambda row: row["source"])


def run_cases(
    engine: AnswerEngine,
    cases: list[EvalCase],
    modes: list[str] | tuple[str, ...],
    repeat: int = 1,
    first_run_temperature: str = "uncontrolled",
    on_result=None,
    *,
    audience: str = "general",
    detail: str = "standard",
) -> list[CaseResult]:
    """Run without installing, unloading, or warming models implicitly.

    Modes are rotated by repetition to expose order effects. All repeat exposure
    is recorded; a second mode seeing the same question is not called unseen.
    """
    if (
        repeat < 1
        or not modes
        or len(set(modes)) != len(modes)
        or not set(modes) <= set(MODES)
    ):
        raise ValueError("invalid repeat or modes")
    if first_run_temperature not in {"uncontrolled", "cold", "warm"}:
        raise ValueError("invalid first-query preparation")
    if audience not in AUDIENCES or detail not in DETAILS:
        raise ValueError("invalid audience or detail")
    results: list[CaseResult] = []
    seen_questions: set[str] = set()
    generation_seen = False
    for repetition in range(repeat):
        offset = repetition % len(modes)
        ordered_modes = list(modes[offset:]) + list(modes[:offset])
        for mode in ordered_modes:
            for case in cases:
                start = time.perf_counter()
                first_evidence = None

                def evidence_ready(_evidence, _started=start) -> None:
                    nonlocal first_evidence
                    if first_evidence is None:
                        first_evidence = time.perf_counter() - _started

                exception_type = ""
                try:
                    options = {"audience": audience, "detail": detail}
                    # Old comparison modes retain their original call contract.
                    answer = engine.answer(
                        case.question,
                        mode=mode,
                        on_evidence=evidence_ready,
                        **(options if mode == "explain" else {}),
                    )
                    if (
                        not isinstance(answer, LabAnswer)
                        or answer.status not in STATUSES
                    ):
                        raise ValueError("engine returned an invalid answer")
                except Exception as exc:  # Preserve the remaining cases; do not log exception payloads.
                    exception_type = type(exc).__name__
                    answer = LabAnswer(
                        status="error",
                        message="Evaluation call failed",
                        issues=("engine_exception",),
                    )
                elapsed = time.perf_counter() - start
                normalized = _normalized(answer.text)
                evidence_by_id = {item.evidence_id: item for item in answer.evidence}
                claims = tuple(
                    _claim_record(claim, evidence_by_id) for claim in answer.claims
                )
                exact_quotes = sum(
                    bool(claim["citations"])
                    and all(ref["exact_quote_match"] for ref in claim["citations"])
                    for claim in claims
                )
                claims_equal_quote = sum(
                    len(claim["citations"]) == 1
                    and bool(claim["citations"][0]["quote"])
                    and claim["text"] == claim["citations"][0]["quote"]
                    for claim in claims
                )
                citations = [ref for claim in claims for ref in claim["citations"]]
                temperature = (
                    f"first_query_{first_run_temperature}"
                    if not results
                    else "subsequent_query_uncontrolled"
                )
                generation_exposure = "no_completed_generation_call"
                if answer.calls:
                    generation_exposure = (
                        "subsequent_generation_uncontrolled"
                        if generation_seen
                        else f"first_generation_{first_run_temperature}"
                    )
                    generation_seen = True
                results.append(
                    CaseResult(
                        case_id=case.id,
                        split=case.split,
                        category=case.category,
                        question=case.question,
                        mode=mode,
                        repeat=repetition + 1,
                        temperature=temperature,
                        generation_exposure=generation_exposure,
                        question_exposure="repeated"
                        if case.question in seen_questions
                        else "first_seen",
                        expected_status=case.expected_status,
                        status=answer.status,
                        answer=answer.text,
                        elapsed_seconds=elapsed,
                        first_evidence_seconds=first_evidence,
                        timings=answer.timings,
                        calls=answer.calls,
                        route=answer.route,
                        issues=answer.issues,
                        claims=claims,
                        evidence_sources=tuple(
                            sorted({item.source_name for item in answer.evidence})
                        ),
                        gold_sources=case.gold_sources,
                        missing_terms=tuple(
                            term
                            for term in case.required_terms
                            if _normalized(term) not in normalized
                        ),
                        forbidden_terms_found=tuple(
                            term
                            for term in case.forbidden_terms
                            if _normalized(term) in normalized
                        ),
                        required_term_count=len(case.required_terms),
                        exact_quote_count=exact_quotes,
                        claim_equals_quote_count=claims_equal_quote,
                        status_match=answer.status in case.expected_status,
                        truncated="truncated" in answer.issues
                        or any(
                            call.get("done_reason") == "length" for call in answer.calls
                        ),
                        exception_type=exception_type,
                        audience=audience,
                        detail=detail,
                        citation_count=len(citations),
                        exact_citation_count=sum(
                            ref["exact_quote_match"] for ref in citations
                        ),
                        review_requirements=case.review_requirements,
                        presentation_options_applied=mode == "explain",
                    )
                )
                seen_questions.add(case.question)
                if on_result is not None:
                    on_result(results[-1])
    return results


def _claim_record(claim: Claim, evidence_by_id: dict[str, Evidence]) -> dict[str, Any]:
    """Preserve every reference and its resolvable provenance, never entailment.

    Legacy single-source claims use their old reference fields. For new claims,
    ``references`` is authoritative: the legacy alias must not double-count the
    first citation or hide a missing second one.
    """
    references = claim.references
    citations = []
    for reference in references:
        evidence = evidence_by_id.get(reference.evidence_id)
        citations.append(
            {
                "evidence_id": reference.evidence_id,
                "quote": reference.quote,
                "source_name": evidence.source_name if evidence else None,
                "page_number": evidence.page_number if evidence else None,
                "content_hash": evidence.content_hash if evidence else None,
                "version_id": evidence.version_id if evidence else None,
                "exact_quote_match": bool(
                    reference.quote and evidence and reference.quote in evidence.text
                ),
            }
        )
    return {**asdict(claim), "citations": citations}


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _latency(values: list[float]) -> dict[str, int | float | None]:
    """Nearest-rank p95; n<20 is explicitly a small descriptive sample."""
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
        "p95_method": "nearest_rank",
        "small_sample": len(values) < 20,
    }


def _call_totals(rows: list[CaseResult], key: str) -> list[float]:
    """Keep missing API metrics missing rather than silently turning them into zero."""
    totals = []
    for row in rows:
        values = [call.get(key) for call in row.calls]
        if values and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in values
        ):
            totals.append(sum(values))
    return totals


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    by_mode: dict[str, list[CaseResult]] = defaultdict(list)
    for row in results:
        by_mode[row.mode].append(row)
    summaries = {}
    for mode, rows in by_mode.items():
        answerable = [row for row in rows if "answered" in row.expected_status]
        unanswerable = [row for row in rows if "answered" not in row.expected_status]
        answered = [row for row in rows if row.status == "answered"]
        lexical = [row for row in answerable if row.required_term_count]
        claims = sum(len(row.claims) for row in answered)
        citations = sum(row.citation_count for row in answered)
        latency_groups: dict[str, list[CaseResult]] = defaultdict(list)
        for row in rows:
            latency_groups[
                f"{row.temperature}/{row.generation_exposure}/{row.question_exposure}/{row.status}"
            ].append(row)
        summaries[mode] = {
            "attempts": len(rows),
            "unique_cases": len({row.case_id for row in rows}),
            "audience_detail_settings": [
                {"audience": audience, "detail": detail}
                for audience, detail in sorted(
                    {(row.audience, row.detail) for row in rows}
                )
            ],
            "answered_fraction": _rate(len(answered), len(rows)),
            "status_agreement": _rate(sum(row.status_match for row in rows), len(rows)),
            "false_answer_rate": _rate(
                sum(row.status == "answered" for row in unanswerable), len(unanswerable)
            ),
            "false_refusal_rate": _rate(
                sum(
                    row.status in {"refused", "clarify", "needs_review"}
                    for row in answerable
                ),
                len(answerable),
            ),
            "answerable_error_rate": _rate(
                sum(row.status == "error" for row in answerable), len(answerable)
            ),
            "error_rate": _rate(sum(row.status == "error" for row in rows), len(rows)),
            "truncation_rate": _rate(sum(row.truncated for row in rows), len(rows)),
            "required_terms_present_and_no_forbidden_terms": _rate(
                sum(
                    row.status == "answered"
                    and not row.missing_terms
                    and not row.forbidden_terms_found
                    for row in lexical
                ),
                len(lexical),
            ),
            "forbidden_terms_any_status": _rate(
                sum(bool(row.forbidden_terms_found) for row in rows), len(rows)
            ),
            "exact_quote_matches": _rate(
                sum(row.exact_quote_count for row in answered), claims
            ),
            "exact_citation_matches": _rate(
                sum(row.exact_citation_count for row in answered), citations
            ),
            "claim_text_equals_quote": _rate(
                sum(row.claim_equals_quote_count for row in answered), claims
            ),
            "gold_source_recall": _rate(
                sum(
                    len(
                        {Path(name).name for name in row.gold_sources}
                        & {Path(name).name for name in row.evidence_sources}
                    )
                    for row in answerable
                ),
                sum(len(row.gold_sources) for row in answerable),
            ),
            "semantic_correctness": "unknown_requires_manual_review",
            "explanation_quality": "unknown_requires_manual_review",
            "latency_by_preparation_exposure_status": {
                key: {
                    "total": _latency([row.elapsed_seconds for row in group]),
                    "first_evidence": _latency(
                        [
                            row.first_evidence_seconds
                            for row in group
                            if row.first_evidence_seconds is not None
                        ]
                    ),
                    "model_stage_totals": {
                        metric: _latency(_call_totals(group, metric))
                        for metric in (
                            "load_seconds",
                            "prefill_seconds",
                            "decode_seconds",
                            "wall_seconds",
                        )
                    },
                }
                for key, group in latency_groups.items()
            },
            "by_category": {
                category: {
                    "attempts": len(group),
                    "answered": sum(row.status == "answered" for row in group),
                    "status_agreement": _rate(
                        sum(row.status_match for row in group), len(group)
                    ),
                }
                for category in sorted({row.category for row in rows})
                if (group := [row for row in rows if row.category == category])
            },
        }
    return {
        "schema_version": 2,
        "limitations": LIMITATIONS,
        "metric_definitions": {
            "exact_quote_matches": "Claims with at least one citation and every citation "
            "an exact substring of its referenced evidence; denominator is claims.",
            "exact_citation_matches": "Exact-substring citation matches; denominator is "
            "all references, including missing evidence and non-primary references.",
            "claim_text_equals_quote": "Claims identical to their sole nonempty quote; "
            "denominator is claims. Multi-source synthesis is not penalized as incorrect.",
        },
        "manual_review_completed": False,
        "modes": summaries,
    }


def _safe_csv(value: Any) -> str:
    text = str(value)
    return (
        "'" + text
        if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
        else text
    )


def write_report(
    output_dir: Path | str, results: list[CaseResult], manifest: dict[str, Any]
) -> dict[str, Path]:
    """Create a fresh directory and refuse to overwrite any prior evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        name: output_dir / name
        for name in (
            "results.jsonl",
            "report.json",
            "manifest.json",
            "manual_review.csv",
        )
    }
    paths["results.jsonl"].write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    paths["report.json"].write_text(
        json.dumps(summarize_results(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["manifest.json"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    with paths["manual_review.csv"].open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_id",
                "split",
                "category",
                "mode",
                "repeat",
                "audience",
                "detail",
                "presentation_options_applied",
                "question",
                "expected_status",
                "actual_status",
                "answer",
                "claims_and_quotes",
                "gold_sources",
                "retrieved_sources",
                "missing_terms",
                "forbidden_terms_found",
                "issues",
                "case_review_requirements",
                "scientifically_correct_yes_no_unclear",
                "all_claims_supported_yes_no_unclear",
                "meaning_and_conditions_preserved_yes_no_unclear",
                "correct_target_yes_no_unclear",
                "synthesis_supported_yes_no_na",
                "audience_appropriate_yes_no_unclear",
                "useful_explanation_yes_no_unclear",
                "appropriate_refusal_yes_no_na",
                "reviewer",
                "notes",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    _safe_csv(value)
                    for value in [
                        row.case_id,
                        row.split,
                        row.category,
                        row.mode,
                        row.repeat,
                        row.audience,
                        row.detail,
                        row.presentation_options_applied,
                        row.question,
                        "/".join(row.expected_status),
                        row.status,
                        row.answer,
                        json.dumps(row.claims, ensure_ascii=False),
                        ";".join(row.gold_sources),
                        ";".join(row.evidence_sources),
                        ";".join(row.missing_terms),
                        ";".join(row.forbidden_terms_found),
                        ";".join(row.issues),
                        ";".join(row.review_requirements),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                ]
            )
    return paths


def _runtime() -> dict[str, Any]:
    versions = {}
    for name in ("httpx", "numpy", "PyYAML", "streamlit", "PyMuPDF"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "packages": versions,
    }


def _code_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return {
        path.relative_to(project_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((project_root / "src").rglob("*.py"))
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="lab_config.yaml")
    parser.add_argument("--dataset", default="benchmarks/lab_v2/cases.jsonl")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=["quoted", "concise"]
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--audience", choices=AUDIENCES, default="general")
    parser.add_argument("--detail", choices=DETAILS, default="standard")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--output-root", default="outputs/lab-comparisons")
    parser.add_argument(
        "--first-query-state",
        choices=("uncontrolled", "cold", "warm"),
        default="uncontrolled",
        help="Operator declaration, not model preparation. Caches are not reset.",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        parser.error("run-id must be a simple directory name")
    if args.limit is not None and args.limit < 1 or args.repeat < 1:
        parser.error("limit and repeat must be positive")
    cases = load_cases(args.dataset, args.split)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        parser.error("selected split is empty")
    sources = load_corpus(args.dataset, args.split)
    models = args.models or [None]
    if len(set(models)) != len(models):
        parser.error("duplicate model names")
    run_root = Path(args.output_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=False)
    from src.lab.factory import build_lab

    for number, model in enumerate(models, 1):
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", model or "configured")
        model_root = run_root / f"{number:02d}-{label}"
        bundle = build_lab(
            config_path=args.config,
            data_dir=model_root / "index",
            generation_model=model,
        )
        try:
            code_hashes = _code_hashes()
            generation_digest = bundle.client.model_digest
            embedding_digest = bundle.embedding.model_digest
            config_hash = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
            registration_start = time.perf_counter()
            for source in sources:
                bundle.index.register_bytes(
                    source["path"].name,
                    source["path"].read_bytes(),
                    approved=source["approved"],
                    effective_date=source["effective_date"],
                )
            registration_seconds = time.perf_counter() - registration_start
            print(
                f"{label}: {len(cases)} cases x {len(args.modes)} modes x {args.repeat} repeat; {args.split}",
                flush=True,
            )
            # Keep completed rows recoverable if a long comparison is interrupted.
            with (model_root / "progress.jsonl").open(
                "x", encoding="utf-8"
            ) as progress:

                def checkpoint(row):
                    progress.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    progress.flush()
                    print(
                        f"  {row.case_id} {row.mode}: {row.status} {row.elapsed_seconds:.2f}s",
                        flush=True,
                    )

                results = run_cases(
                    bundle.engine,
                    cases,
                    args.modes,
                    args.repeat,
                    args.first_query_state,
                    on_result=checkpoint,
                    audience=args.audience,
                    detail=args.detail,
                )
            settings_snapshot = asdict(bundle.settings)
            settings_snapshot["storage"] = {"root": "<isolated per-run index>"}
            settings_snapshot["ollama"]["generation_model"] = bundle.client.model_name
            manifest = {
                "schema_version": 2,
                "created_utc": datetime.now(UTC).isoformat(),
                "split": args.split,
                "case_ids": [case.id for case in cases],
                "dataset_sha256": hashlib.sha256(
                    Path(args.dataset).read_bytes()
                ).hexdigest(),
                "config_sha256": config_hash,
                "code_sha256": code_hashes,
                "code_changed_during_run": code_hashes != _code_hashes(),
                "config_changed_during_run": config_hash
                != hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                "effective_config": settings_snapshot,
                "generation_model": {
                    "name": bundle.client.model_name,
                    "digest": generation_digest,
                },
                "embedding_model": {
                    "name": bundle.embedding.model_name,
                    "digest": embedding_digest,
                },
                "model_weights_changed_during_run": generation_digest
                != bundle.client.model_digest
                or embedding_digest != bundle.embedding.model_digest,
                "index_registration_seconds": registration_seconds,
                "index_revision": bundle.index.revision,
                "corpus_manifest_sha256": hashlib.sha256(
                    (Path(args.dataset).parent / "corpus_manifest.json").read_bytes()
                ).hexdigest(),
                "generation_model_override": model,
                "modes": args.modes,
                "audience": args.audience,
                "detail": args.detail,
                "presentation_options_apply_to": ["explain"],
                "repeat": args.repeat,
                "first_query_preparation_declared": args.first_query_state,
                "runtime": _runtime(),
                "sources": [
                    {key: value for key, value in source.items() if key != "path"}
                    for source in sources
                ],
                "observed_model_digests": sorted(
                    {
                        str(call["digest"])
                        for result in results
                        for call in result.calls
                        if call.get("digest")
                    }
                ),
                "limitations": LIMITATIONS,
                "holdout_use": "exploratory_requires_fresh_external_validation"
                if args.split == "holdout"
                else "development",
                "dataset_use": "exploratory_after_exposure"
                if args.split == "holdout"
                else "exposed_development_only",
            }
            paths = write_report(model_root / "report", results, manifest)
            print(paths["report.json"], flush=True)
        finally:
            bundle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
