from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.lab.contracts import Claim, Evidence, LabAnswer, SourceCitation
from src.lab.evaluate import (
    EvalCase,
    load_cases,
    load_corpus,
    main,
    run_cases,
    summarize_results,
    write_report,
)

FIXTURE = Path(__file__).parents[1] / "benchmarks" / "lab_v2" / "cases.jsonl"
EXPLANATION_FIXTURE = (
    Path(__file__).parents[1] / "benchmarks" / "explanation_dev" / "cases.jsonl"
)


def sample_case(**kwargs):
    defaults = dict(
        id="case-1",
        split="dev",
        category="numeric",
        question="何席ありますか？",
        gold_sources=("dev/seats.txt",),
        required_terms=("48席",),
        forbidden_terms=("50席",),
        expected_status=("answered",),
    )
    return EvalCase(**(defaults | kwargs))


class FakeEngine:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
        self.options = []

    def answer(self, question, mode, on_evidence=None, **options):
        self.calls.append((question, mode))
        self.options.append(options)
        result = next(self.answers)
        if isinstance(result, Exception):
            raise result
        if on_evidence and result.evidence:
            on_evidence(result.evidence)
        return result


def test_fixture_is_large_source_disjoint_and_covers_failure_types():
    all_cases = load_cases(FIXTURE)
    dev = load_cases(FIXTURE, "dev")
    holdout = load_cases(FIXTURE, "holdout")
    assert len(dev) >= 30 and len(holdout) >= 20
    assert len(all_cases) == len(dev) + len(holdout)
    assert {case.category for case in dev} == {case.category for case in holdout}
    dev_names = {item["path"].name for item in load_corpus(FIXTURE, "dev")}
    holdout_names = {item["path"].name for item in load_corpus(FIXTURE, "holdout")}
    assert not dev_names & holdout_names
    dev_hashes = {item["sha256"] for item in load_corpus(FIXTURE, "dev")}
    holdout_hashes = {item["sha256"] for item in load_corpus(FIXTURE, "holdout")}
    assert not dev_hashes & holdout_hashes
    for split in ("dev", "holdout"):
        sources = load_corpus(FIXTURE, split)
        assert any(not item["approved"] for item in sources)
        assert any(item["effective_date"] == "2099-01-01" for item in sources)


def write_fixture(tmp_path, **changes):
    source = tmp_path / "corpus" / "dev" / "seats.txt"
    source.parent.mkdir(parents=True)
    source.write_text("48席", encoding="utf-8")
    raw = {
        "id": "case-1",
        "split": "dev",
        "category": "numeric",
        "question": "何席ありますか？",
        "gold_sources": ["dev/seats.txt"],
        "required_terms": ["48席"],
        "forbidden_terms": ["50席"],
        "expected_status": ["answered"],
    } | changes
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"gold_sources": ["holdout/seats.txt"]},
        {"gold_sources": ["dev/../../outside.txt"]},
        {"gold_sources": ["dev/missing.txt"]},
        {"required_terms": "48席"},
        {"required_terms": []},
        {"forbidden_terms": ["４８席"]},
        {"expected_status": ["answered", "refused"]},
        {"expected_status": ["error"]},
        {"expected_status": ["unknown"]},
        {"category": "unregistered"},
        {"split": "test"},
        {"question": " "},
    ],
)
def test_loader_rejects_invalid_or_ambiguous_labels(tmp_path, changes):
    path = write_fixture(tmp_path, **changes)
    with pytest.raises(ValueError):
        load_cases(path)


def test_loader_validates_other_split_and_rejects_duplicate_questions(tmp_path):
    path = write_fixture(tmp_path)
    line = path.read_text(encoding="utf-8")
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path, split="holdout")


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = write_fixture(tmp_path)
    line = path.read_text(encoding="utf-8").replace(
        '"id": "case-1"', '"id": "case-1", "id": "case-2"'
    )
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_cases(path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"source": "../private.txt", "approved": True, "effective_date": None},
        {"source": "dev/seats.txt", "approved": "yes", "effective_date": None},
        {"source": "dev/seats.txt", "approved": True, "effective_date": 20260901},
        {"source": "dev/seats.txt", "approved": True, "effective_date": "2026-9-1"},
    ],
)
def test_corpus_manifest_rejects_invalid_metadata(tmp_path, metadata):
    path = write_fixture(tmp_path)
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "self_authored": True, "documents": [metadata]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_corpus(path, "dev")


def test_corpus_manifest_rejects_unlisted_file(tmp_path):
    path = write_fixture(tmp_path)
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "self_authored": True,
                "documents": [
                    {
                        "source": "dev/seats.txt",
                        "approved": True,
                        "effective_date": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "corpus" / "dev" / "unlisted.txt").write_text(
        "not admitted", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest differ"):
        load_corpus(path, "dev")


def test_metrics_distinguish_status_lexical_provenance_and_semantic_truth():
    evidence = Evidence("e1", "定員は48席です。", "seats.txt", 1, "hash")
    # Exact quote exists but the associated generated claim contradicts it.
    answer = LabAnswer(
        "answered",
        "",
        claims=(Claim("定員は50席です。", "e1", "定員は48席です。"),),
        evidence=(evidence,),
    )
    results = run_cases(FakeEngine([answer]), [sample_case()], ["concise"])
    result = results[0]
    assert result.status_match
    assert result.exact_quote_count == 1
    assert result.claim_equals_quote_count == 0
    assert result.missing_terms == ("48席",)
    assert result.forbidden_terms_found == ("50席",)
    assert result.first_evidence_seconds is not None
    report = summarize_results(results)["modes"]["concise"]
    assert report["exact_quote_matches"]["rate"] == 1
    assert report["required_terms_present_and_no_forbidden_terms"]["rate"] == 0
    assert report["semantic_correctness"] == "unknown_requires_manual_review"
    assert report["false_answer_rate"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }


def test_missing_evidence_and_empty_quote_never_count_as_verified():
    answer = LabAnswer(
        "answered",
        "",
        claims=(Claim("48席", "missing", "48席"), Claim("48席", "also-missing", "")),
    )
    result = run_cases(FakeEngine([answer]), [sample_case()], ["quoted"])[0]
    assert result.exact_quote_count == 0


def test_errors_truncations_and_refusals_have_distinct_denominators():
    cases = [
        sample_case(id="good-1"),
        sample_case(id="good-2", question="席数は？"),
        sample_case(
            id="bad-1",
            question="明日の天気は？",
            expected_status=("refused",),
            required_terms=(),
            gold_sources=(),
        ),
    ]
    engine = FakeEngine(
        [
            LabAnswer("needs_review", "", issues=("truncated",)),
            RuntimeError("do not expose secret payload"),
            LabAnswer("answered", "晴れです"),
        ]
    )
    results = run_cases(engine, cases, ["concise"])
    report = summarize_results(results)["modes"]["concise"]
    assert report["false_answer_rate"] == {"numerator": 1, "denominator": 1, "rate": 1}
    assert report["false_refusal_rate"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert report["answerable_error_rate"]["rate"] == 0.5
    assert report["truncation_rate"]["numerator"] == 1
    assert results[1].exception_type == "RuntimeError"
    assert "secret" not in results[1].answer


def test_first_query_and_repeated_exposure_are_not_pooled_as_warm():
    cases = [sample_case()]
    engine = FakeEngine([LabAnswer("refused", "") for _ in range(4)])
    results = run_cases(
        engine, cases, ["baseline", "quoted"], repeat=2, first_run_temperature="cold"
    )
    assert [row.mode for row in results] == ["baseline", "quoted", "quoted", "baseline"]
    assert results[0].temperature == "first_query_cold"
    assert all(
        row.temperature == "subsequent_query_uncontrolled" for row in results[1:]
    )
    assert [row.question_exposure for row in results] == [
        "first_seen",
        "repeated",
        "repeated",
        "repeated",
    ]
    report = summarize_results(results)
    latency = report["modes"]["baseline"]["latency_by_preparation_exposure_status"]
    assert len(latency) == 2
    assert all(group["total"]["n"] == 1 for group in latency.values())


def test_review_sheet_is_always_unfilled_and_safe_to_open_in_spreadsheet(tmp_path):
    results = run_cases(
        FakeEngine([LabAnswer("answered", '=HYPERLINK("bad")')]),
        [sample_case()],
        ["concise"],
    )
    paths = write_report(tmp_path / "new-report", results, {"self_authored": True})
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    assert report["manual_review_completed"] is False
    with paths["manual_review.csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["answer"].startswith("'=")
    assert rows[0]["scientifically_correct_yes_no_unclear"] == ""
    with pytest.raises(FileExistsError):
        write_report(tmp_path / "new-report", results, {})


def test_latency_uses_nearest_rank_p95_and_finite_samples():
    row = run_cases(
        FakeEngine([LabAnswer("answered", "48席")]), [sample_case()], ["quoted"]
    )[0]
    results = [
        replace(row, case_id=f"c-{index}", elapsed_seconds=float(index))
        for index in range(1, 21)
    ]
    report = summarize_results(results)["modes"]["quoted"]
    group = next(iter(report["latency_by_preparation_exposure_status"].values()))
    assert group["total"]["median_seconds"] == 10.5
    assert group["total"]["p95_seconds"] == 19
    assert group["total"]["small_sample"] is False


def test_missing_model_metrics_are_not_converted_to_zero():
    answer = LabAnswer(
        "answered",
        "48席",
        calls=({"load_seconds": 3.0, "prefill_seconds": "not_reported"},),
    )
    results = run_cases(FakeEngine([answer]), [sample_case()], ["quoted"])
    report = summarize_results(results)["modes"]["quoted"]
    group = next(iter(report["latency_by_preparation_exposure_status"].values()))
    assert group["model_stage_totals"]["load_seconds"]["median_seconds"] == 3.0
    assert group["model_stage_totals"]["prefill_seconds"]["median_seconds"] is None


@pytest.mark.parametrize(
    "mode, repeat",
    [(["unknown"], 1), (["quoted", "quoted"], 1), ([], 1), (["quoted"], 0)],
)
def test_run_rejects_invalid_experiment_settings(mode, repeat):
    with pytest.raises(ValueError):
        run_cases(FakeEngine([]), [sample_case()], mode, repeat)


def test_explanation_fixture_is_exposed_development_not_old_holdout():
    cases = load_cases(EXPLANATION_FIXTURE)
    assert len(cases) == 9
    assert {case.split for case in cases} == {"dev"}
    assert load_cases(EXPLANATION_FIXTURE, "holdout") == []
    assert all(case.review_requirements for case in cases)
    assert {
        "synthesis",
        "meaning_retention",
        "condition_retention",
        "wrong_target",
        "missing_fact",
        "contradiction",
    } <= {case.category for case in cases}
    new_sources = load_corpus(EXPLANATION_FIXTURE, "dev")
    old_sources = load_corpus(FIXTURE, "dev") + load_corpus(FIXTURE, "holdout")
    assert len(new_sources) == 6
    assert not {source["sha256"] for source in new_sources} & {
        source["sha256"] for source in old_sources
    }
    manifest = json.loads(
        EXPLANATION_FIXTURE.with_name("corpus_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["research_use"] == "exposed_development_only"
    assert manifest["expert_validated"] is False


def test_legacy_fixture_still_loads_without_optional_review_requirements(tmp_path):
    assert load_cases(write_fixture(tmp_path))[0].review_requirements == ()


@pytest.mark.parametrize("requirements", ["not a list", [""], ["same", "same"]])
def test_invalid_review_requirements_are_rejected(tmp_path, requirements):
    with pytest.raises(ValueError, match="review_requirements"):
        load_cases(write_fixture(tmp_path, review_requirements=requirements))


@pytest.mark.parametrize("audience", ["general", "child", "staff"])
@pytest.mark.parametrize("detail", ["short", "standard"])
def test_explanation_presentation_options_are_forwarded_and_recorded(audience, detail):
    engine = FakeEngine([LabAnswer("refused", "資料にありません")])
    row = run_cases(
        engine, [sample_case()], ["explain"], audience=audience, detail=detail
    )[0]
    assert engine.options == [{"audience": audience, "detail": detail}]
    assert (row.audience, row.detail) == (audience, detail)
    assert row.presentation_options_applied
    report = summarize_results([row])["modes"]["explain"]
    assert report["audience_detail_settings"] == [
        {"audience": audience, "detail": detail}
    ]
    assert report["explanation_quality"] == "unknown_requires_manual_review"


def test_old_modes_keep_their_original_call_contract():
    engine = FakeEngine([LabAnswer("refused", "資料にありません")])
    row = run_cases(
        engine, [sample_case()], ["quoted"], audience="child", detail="short"
    )[0]
    assert engine.options == [{}]
    assert not row.presentation_options_applied


@pytest.mark.parametrize("options", [{"audience": "unknown"}, {"detail": "long"}])
def test_invalid_presentation_options_fail_before_engine_call(options):
    engine = FakeEngine([])
    with pytest.raises(ValueError, match="audience or detail"):
        run_cases(engine, [sample_case()], ["explain"], **options)
    assert not engine.calls


def test_every_reference_is_counted_and_saved_with_source_provenance(tmp_path):
    first = Evidence(
        "e1", "定員は48席です。", "seats.txt", 1, "hash-1", version_id="v1"
    )
    second = Evidence(
        "e2", "整理券が必要です。", "tickets.txt", 3, "hash-2", version_id="v2"
    )
    # The text is deliberately false: correct quotation provenance is NOT entailment.
    claim = Claim(
        "50席あり、整理券は不要です。",
        "e1",
        first.text,
        citations=(SourceCitation("e1", first.text), SourceCitation("e2", second.text)),
    )
    answer = LabAnswer("answered", "", claims=(claim,), evidence=(first, second))
    row = run_cases(FakeEngine([answer]), [sample_case()], ["explain"])[0]
    assert row.citation_count == 2
    assert row.exact_citation_count == 2
    assert row.exact_quote_count == 1
    assert row.claim_equals_quote_count == 0
    report = summarize_results([row])["modes"]["explain"]
    assert report["exact_citation_matches"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1,
    }
    assert report["exact_quote_matches"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1,
    }
    assert report["semantic_correctness"] == "unknown_requires_manual_review"
    paths = write_report(tmp_path / "report", [row], {})
    saved = json.loads(paths["results.jsonl"].read_text(encoding="utf-8"))
    assert saved["claims"][0]["citations"] == [
        {
            "evidence_id": "e1",
            "quote": first.text,
            "source_name": "seats.txt",
            "page_number": 1,
            "content_hash": "hash-1",
            "version_id": "v1",
            "exact_quote_match": True,
        },
        {
            "evidence_id": "e2",
            "quote": second.text,
            "source_name": "tickets.txt",
            "page_number": 3,
            "content_hash": "hash-2",
            "version_id": "v2",
            "exact_quote_match": True,
        },
    ]


@pytest.mark.parametrize(
    "second_reference",
    [
        SourceCitation("missing", "整理券が必要です。"),
        SourceCitation("e2", "整理券は不要です。"),
        SourceCitation("e2", ""),
    ],
)
def test_bad_secondary_citation_cannot_hide_behind_valid_first_reference(
    second_reference,
):
    first = Evidence("e1", "定員は48席です。", "seats.txt", 1, "hash-1")
    second = Evidence("e2", "整理券が必要です。", "tickets.txt", 2, "hash-2")
    answer = LabAnswer(
        "answered",
        "",
        evidence=(first, second),
        claims=(
            Claim(
                "48席です。",
                "e1",
                first.text,
                citations=(SourceCitation("e1", first.text), second_reference),
            ),
        ),
    )
    row = run_cases(FakeEngine([answer]), [sample_case()], ["explain"])[0]
    assert row.exact_quote_count == 0
    assert (row.exact_citation_count, row.citation_count) == (1, 2)
    assert row.claims[0]["citations"][1]["exact_quote_match"] is False
    report = summarize_results([row])["modes"]["explain"]
    assert report["exact_citation_matches"]["rate"] == 0.5
    assert report["exact_quote_matches"]["rate"] == 0


def test_review_sheet_carries_requirements_and_blank_explanation_judgments(tmp_path):
    requirements = ("条件を省かない", "対象を取り違えない")
    case = sample_case(review_requirements=requirements)
    row = run_cases(
        FakeEngine([LabAnswer("refused", "資料にありません")]),
        [case],
        ["explain"],
        audience="child",
        detail="short",
    )[0]
    paths = write_report(tmp_path / "report", [row], {})
    with paths["manual_review.csv"].open(encoding="utf-8-sig", newline="") as stream:
        review = list(csv.DictReader(stream))[0]
    assert None not in review  # Every header and value is aligned.
    assert review["case_review_requirements"] == ";".join(requirements)
    assert (review["audience"], review["detail"]) == ("child", "short")
    assert review["presentation_options_applied"] == "True"
    judgments = [key for key in review if "_yes_no_" in key]
    assert len(judgments) == 8
    assert all(review[key] == "" for key in judgments)


def test_cli_records_explanation_settings_and_exposure_without_real_models(
    tmp_path, monkeypatch
):
    from src.lab import factory

    @dataclass
    class Settings:
        storage: dict
        ollama: dict

    engine = FakeEngine([LabAnswer("refused", "資料にありません")])
    registered = []
    closed = []

    def build_fake(**kwargs):
        kwargs["data_dir"].mkdir(parents=True)
        return SimpleNamespace(
            engine=engine,
            client=SimpleNamespace(
                model_name="fake-model", model_digest="fake-generation"
            ),
            embedding=SimpleNamespace(
                model_name="fake-embedding", model_digest="fake-embedding"
            ),
            settings=Settings(storage={}, ollama={}),
            index=SimpleNamespace(
                revision="fake-revision",
                register_bytes=lambda *args, **kwargs: registered.append(
                    (args, kwargs)
                ),
            ),
            close=lambda: closed.append(True),
        )

    monkeypatch.setattr(factory, "build_lab", build_fake)
    config = tmp_path / "config.yaml"
    config.write_text(
        "# No model or network needed for the fake CLI test\n", encoding="utf-8"
    )
    output = tmp_path / "runs"
    assert (
        main(
            [
                "--config",
                str(config),
                "--dataset",
                str(EXPLANATION_FIXTURE),
                "--split",
                "dev",
                "--modes",
                "explain",
                "--audience",
                "child",
                "--detail",
                "short",
                "--limit",
                "1",
                "--run-id",
                "fake-cli",
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    report_dir = output / "fake-cli" / "01-configured" / "report"
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((report_dir / "results.jsonl").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert (manifest["audience"], manifest["detail"]) == ("child", "short")
    assert manifest["presentation_options_apply_to"] == ["explain"]
    assert manifest["dataset_use"] == "exposed_development_only"
    assert result["review_requirements"]
    assert result["presentation_options_applied"] is True
    assert len(registered) == 6
    assert closed == [True]
    assert engine.options == [{"audience": "child", "detail": "short"}]
