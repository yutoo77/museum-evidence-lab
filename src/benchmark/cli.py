"""隔離領域で公開デモRAG Benchmarkを実行するCLI。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.benchmark.dataset import load_benchmark_cases
from src.benchmark.metrics import compute_metrics
from src.benchmark.report import write_benchmark_artifacts
from src.benchmark.runner import run_benchmark
from src.benchmark.services import build_isolated_benchmark_services
from src.config import load_config
from src.demo_data import DEMO_DOCUMENT_FILENAMES, register_demo_documents
from src.models import AnswerStatus


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError(
            "run idは英数字で始まる80文字以内の英数字・dot・hyphen・underscoreです。"
        )
    return value


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="公開デモ資料を隔離領域へ登録してRAG smoke benchmarkを実行します。"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="benchmarks/rag_v1/cases.jsonl")
    parser.add_argument("--sample-dir", default="sample_docs")
    parser.add_argument("--output-root", default="outputs/benchmarks")
    parser.add_argument("--run-id")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--distance-threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = datetime.now(UTC)
    generated_id = started.strftime("smoke-%Y%m%dT%H%M%SZ")
    try:
        run_id = _safe_run_id(args.run_id or generated_id)
        base_config = load_config(args.config)
        dataset_path = Path(args.dataset).resolve()
        sample_dir = Path(args.sample_dir).resolve()
        output_root = Path(args.output_root).resolve()
        run_directory = output_root / run_id
        if run_directory.exists():
            raise FileExistsError(f"同名のrunが既に存在します: {run_directory}")
        run_directory.mkdir(parents=True, exist_ok=False)

        cases = load_benchmark_cases(dataset_path)
        if {case.split.value for case in cases} != {"smoke"}:
            raise ValueError("このCLIの初期版はsmoke splitだけを実行できます。")
        services = build_isolated_benchmark_services(base_config, run_directory)
        registration = register_demo_documents(
            services.registration_service, sample_dir
        )
        if registration.errors:
            raise RuntimeError("資料登録に失敗しました: " + " / ".join(registration.errors))

        top_k = (
            args.top_k if args.top_k is not None else base_config.retrieval.top_k
        )
        threshold = (
            args.distance_threshold
            if args.distance_threshold is not None
            else base_config.retrieval.distance_threshold
        )
        if top_k < 1:
            raise ValueError("top-kは1以上である必要があります。")
        if not 0 <= threshold <= 2:
            raise ValueError("distance thresholdは0以上2以下である必要があります。")
        results = run_benchmark(
            services.qa_service,
            cases,
            top_k=top_k,
            distance_threshold=threshold,
        )
        metrics = compute_metrics(results, retrieval_k=top_k)
        complete = not any(
            result.actual_status == AnswerStatus.ERROR for result in results
        )
        finished = datetime.now(UTC)
        corpus_hashes = {
            filename: _sha256(sample_dir / filename)
            for filename in DEMO_DOCUMENT_FILENAMES
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "complete": complete,
            "research_use": "smoke_only",
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "git": _git_state(base_config.project_root),
            "dataset": {
                "path": str(dataset_path),
                "sha256": _sha256(dataset_path),
                "case_count": len(cases),
            },
            "corpus_sha256": corpus_hashes,
            "configuration": {
                "chunk_size": base_config.chunking.chunk_size,
                "chunk_overlap": base_config.chunking.chunk_overlap,
                "top_k": top_k,
                "distance_threshold": threshold,
                "embedding_model": base_config.ollama.embedding_model,
                "generation_model": base_config.ollama.generation_model,
                "ollama_base_url": base_config.ollama.base_url,
            },
            "storage": {
                "run_directory": str(run_directory),
                "isolated_from_application_data": True,
            },
        }
        artifacts = write_benchmark_artifacts(
            run_directory, results, metrics, manifest
        )
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "complete": complete,
                    "status_accuracy": metrics.status_accuracy,
                    "false_answer_rate": metrics.false_answer_rate,
                    "false_refusal_rate": metrics.false_refusal_rate,
                    "retrieval_hit_rate": metrics.retrieval_hit_rate,
                    "p95_elapsed_seconds": metrics.p95_elapsed_seconds,
                    "summary": str(artifacts.summary_json),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if complete else 2
    except Exception as exc:
        print(f"Benchmarkを実行できませんでした: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
