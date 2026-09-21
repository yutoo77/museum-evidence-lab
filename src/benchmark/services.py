"""既存の運用Dataへ触れないBenchmark専用Service構成。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.config import AppConfig, StorageSection
from src.runtime import ApplicationServices, build_services_from_config


def isolated_benchmark_config(
    base_config: AppConfig, run_directory: str | Path
) -> AppConfig:
    """run directory配下だけへ永続化する設定を返す。"""

    run_root = Path(run_directory).resolve()
    project_root = base_config.project_root.resolve()
    if run_root == project_root:
        raise ValueError("Benchmarkの保存先にProject root自体は指定できません。")

    data_root = run_root / "data"
    storage = StorageSection(
        documents_dir=data_root / "documents",
        chroma_dir=data_root / "chroma",
        database_path=data_root / "benchmark.db",
        log_dir=run_root / "logs",
        interaction_log_path=run_root / "logs" / "interactions.jsonl",
        collection_name="benchmark_documents",
    )
    return replace(base_config, storage=storage)


def build_isolated_benchmark_services(
    base_config: AppConfig, run_directory: str | Path
) -> ApplicationServices:
    config = isolated_benchmark_config(base_config, run_directory)
    config.ensure_directories()
    return build_services_from_config(config)
