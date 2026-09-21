from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

MANIFEST_PATH = Path("benchmarks/rag_v1/baseline_manifest.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_manifest_marks_the_dataset_as_smoke_only():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["research_use"] == "smoke_only"
    assert manifest["rag_config"]["distance_threshold"] == 0.75


def test_baseline_manifest_hashes_match_the_frozen_public_artifacts():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = manifest["sha256"]
    files = {
        "config_yaml": Path("config.yaml"),
        "environment_snapshot": Path("benchmarks/rag_v1/environment.snapshot"),
        "legacy_evaluation_questions_json": Path("evaluation_questions.json"),
        "benchmark_cases_jsonl": Path("benchmarks/rag_v1/cases.jsonl"),
        "benchmark_case_schema_json": Path("benchmarks/rag_v1/case.schema.json"),
    }
    corpus = {
        "moon_phase_txt": Path("sample_docs/moon_phase.txt"),
        "solar_system_txt": Path("sample_docs/solar_system.txt"),
        "black_hole_txt": Path("sample_docs/black_hole.txt"),
        "planetarium_guide_txt": Path("sample_docs/planetarium_guide.txt"),
        "museum_faq_txt": Path("sample_docs/museum_faq.txt"),
    }

    assert {name: _sha256(path) for name, path in files.items()} == {
        name: recorded[name] for name in files
    }
    assert {name: _sha256(path) for name, path in corpus.items()} == recorded["corpus"]


def test_initial_result_is_explicitly_smoke_and_uses_the_frozen_dataset():
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    result = json.loads(
        Path("benchmarks/rag_v1/initial_smoke_result.json").read_text(encoding="utf-8")
    )

    assert result["research_use"] == "smoke_only"
    assert result["dataset_sha256"] == manifest["sha256"]["benchmark_cases_jsonl"]
    assert result["metrics"]["total"] == 9
