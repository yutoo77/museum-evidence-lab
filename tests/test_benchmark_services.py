from __future__ import annotations

from src.benchmark.services import isolated_benchmark_config


def test_isolated_config_keeps_all_runtime_storage_inside_run_directory(
    test_config, tmp_path
):
    run_directory = tmp_path / "run-one"

    isolated = isolated_benchmark_config(test_config, run_directory)

    assert isolated.storage.documents_dir == run_directory / "data" / "documents"
    assert isolated.storage.chroma_dir == run_directory / "data" / "chroma"
    assert isolated.storage.database_path == run_directory / "data" / "benchmark.db"
    assert isolated.storage.interaction_log_path == run_directory / "logs" / "interactions.jsonl"
    assert isolated.storage.documents_dir != test_config.storage.documents_dir
    assert isolated.ollama == test_config.ollama


def test_project_root_itself_cannot_be_used_as_benchmark_run_directory(test_config):
    try:
        isolated_benchmark_config(test_config, test_config.project_root)
    except ValueError as exc:
        assert "Project root" in str(exc)
    else:
        raise AssertionError("ValueError was not raised")
