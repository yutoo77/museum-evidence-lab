"""Strict configuration for the isolated local comparison application."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.lab.client import validate_allowed_models, validate_base_url


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://127.0.0.1:11435"
    embedding_model: str = "embeddinggemma"
    generation_model: str = "qwen3:1.7b"
    allowed_models: tuple[str, ...] = (
        "embeddinggemma",
        "qwen3:1.7b",
        "qwen3.5:2b",
        "qwen3:4b-instruct-2507-q4_K_M",
    )
    context_length: int = 4096
    timeout_seconds: float = 90


@dataclass(frozen=True)
class StorageSettings:
    root: Path = Path("data/lab")


@dataclass(frozen=True)
class LoggingSettings:
    enabled: bool = False
    retention_days: int = 7


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int = 800
    overlap: int = 120
    max_evidence: int = 3
    max_context_chars: int = 1800
    threshold: float = 0.75


@dataclass(frozen=True)
class LabSettings:
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)


def _mapping(value: Any, permitted: set[str], section: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{section}には設定名と値の組を指定してください。")
    unexpected = set(value) - permitted
    if unexpected:
        raise ValueError(
            f"{section}に未対応の設定があります。設定名を確認してください。"
        )
    return value


def _number(
    value: Any, minimum: float, maximum: float, name: str, *, integer: bool = False
):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"{name}は{minimum}〜{maximum}の{'整数' if integer else '数値'}で指定してください。"
        )
    return value


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overriding security options."""


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("設定名は重複のない文字列で指定してください。")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def load_lab_settings(path: str | Path = "lab_config.yaml") -> LabSettings:
    config_path = Path(path).resolve(strict=True)
    try:
        data = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("比較版の設定ファイルを読み取れませんでした。") from exc
    data = _mapping(data, {"ollama", "storage", "logging", "retrieval"}, "設定ファイル")
    defaults = LabSettings()
    ollama = _mapping(
        data.get("ollama", {}), set(OllamaSettings.__dataclass_fields__), "ollama"
    )
    allowed = validate_allowed_models(
        ollama.get("allowed_models", defaults.ollama.allowed_models)
    )
    model_names = {}
    for key in ("embedding_model", "generation_model"):
        name = ollama.get(key, getattr(defaults.ollama, key))
        if not isinstance(name, str) or name not in allowed:
            raise ValueError(
                f"{key}はallowed_modelsに記載したモデルから選んでください。"
            )
        model_names[key] = name
    ollama_settings = OllamaSettings(
        base_url=validate_base_url(ollama.get("base_url", defaults.ollama.base_url)),
        allowed_models=allowed,
        context_length=_number(
            ollama.get("context_length", 4096),
            512,
            32768,
            "context_length",
            integer=True,
        ),
        timeout_seconds=_number(
            ollama.get("timeout_seconds", 90), 1, 600, "timeout_seconds"
        ),
        **model_names,
    )
    storage = _mapping(data.get("storage", {}), {"root"}, "storage")
    root_value = storage.get("root", "data/lab")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("保存先にはプロジェクト内の相対パスを指定してください。")
    relative_root = Path(root_value)
    project_root = config_path.parent.resolve()
    resolved_root = (project_root / relative_root).resolve()
    if (
        relative_root.is_absolute()
        or relative_root.drive
        or ".." in relative_root.parts
        or resolved_root == project_root
        or not resolved_root.is_relative_to(project_root)
    ):
        raise ValueError("保存先は比較版のプロジェクト内に限定してください。")
    logging = _mapping(
        data.get("logging", {}), {"enabled", "retention_days"}, "logging"
    )
    enabled = logging.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("logging.enabledにはtrueまたはfalseを指定してください。")
    logging_settings = LoggingSettings(
        enabled=enabled,
        retention_days=_number(
            logging.get("retention_days", 7), 1, 365, "retention_days", integer=True
        ),
    )
    retrieval = _mapping(
        data.get("retrieval", {}),
        set(RetrievalSettings.__dataclass_fields__),
        "retrieval",
    )
    retrieval_settings = RetrievalSettings(
        chunk_size=_number(
            retrieval.get("chunk_size", 800), 100, 4000, "chunk_size", integer=True
        ),
        overlap=_number(
            retrieval.get("overlap", 120), 0, 2000, "overlap", integer=True
        ),
        max_evidence=_number(
            retrieval.get("max_evidence", 3), 1, 8, "max_evidence", integer=True
        ),
        max_context_chars=_number(
            retrieval.get("max_context_chars", 1800),
            200,
            12000,
            "max_context_chars",
            integer=True,
        ),
        threshold=_number(retrieval.get("threshold", 0.75), 0, 2, "threshold"),
    )
    if retrieval_settings.overlap >= retrieval_settings.chunk_size:
        raise ValueError("overlapはchunk_sizeより小さくしてください。")
    if retrieval_settings.max_context_chars < retrieval_settings.chunk_size:
        raise ValueError("max_context_charsはchunk_size以上にしてください。")
    return LabSettings(
        ollama=ollama_settings,
        storage=StorageSettings(root=resolved_root),
        logging=logging_settings,
        retrieval=retrieval_settings,
    )
