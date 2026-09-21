"""Fail-closed configuration parsing and isolated paths."""

from __future__ import annotations

import pytest

from src.lab.client import LocalModelSecurityError
from src.lab.settings import load_lab_settings


def settings_file(tmp_path, contents):
    path = tmp_path / "lab_config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_minimal_config_defaults_to_isolated_local_models(tmp_path):
    config = load_lab_settings(settings_file(tmp_path, "{}"))
    assert config.ollama.base_url == "http://127.0.0.1:11435"
    assert config.ollama.generation_model == "qwen3:1.7b"
    assert config.storage.root == tmp_path / "data" / "lab"
    assert config.logging.enabled is False
    assert not config.storage.root.exists()


@pytest.mark.parametrize(
    "contents",
    [
        "ollama:\n  base_url: http://external.example:11434",
        "ollama:\n  generation_model: not-allowed:latest",
        "ollama:\n  allowed_models: [qwen3:cloud]",
        "ollama:\n  allowed_models: qwen3:1.7b",
        "ollama:\n  allowed_models: [embeddinggemma, 'embeddinggemma:latest']",
        "ollama:\n  context_length: true",
        "ollama:\n  timeout_seconds: .nan",
        "ollama:\n  base_ur1: http://localhost:11435",
        "storage:\n  root: ../old-app/data",
        "storage:\n  root: .",
        "storage:\n  root: /outside",
        "logging:\n  enabled: 'false'",
        "logging:\n  retention_days: 0",
        "retrieval:\n  chunk_size: 800\n  overlap: 800",
        "retrieval:\n  chunk_size: 1000\n  max_context_chars: 800",
        "retrieval:\n  max_evidence: 0",
        "retrieval:\n  threshold: .inf",
        "ollama:\n  base_url: http://localhost:11435\n  base_url: http://localhost:11434",
        "unknown: true",
        "- wrong-root-shape",
        "",
    ],
)
def test_rejects_unsafe_mistyped_or_inconsistent_config(tmp_path, contents):
    with pytest.raises((ValueError, LocalModelSecurityError)):
        load_lab_settings(settings_file(tmp_path, contents))


def test_valid_config_preserves_explicit_limits(tmp_path):
    path = settings_file(
        tmp_path,
        """
ollama:
  base_url: http://localhost:11435/
  generation_model: qwen3.5:2b
  timeout_seconds: 45
storage:
  root: data/comparison
logging:
  enabled: true
  retention_days: 3
retrieval:
  max_evidence: 2
  threshold: 0.6
""",
    )
    config = load_lab_settings(path)
    assert config.ollama.base_url == "http://127.0.0.1:11435"
    assert config.ollama.generation_model == "qwen3.5:2b"
    assert config.logging.enabled is True
    assert config.logging.retention_days == 3
    assert config.retrieval.threshold == 0.6
    assert config.storage.root == tmp_path / "data/comparison"
