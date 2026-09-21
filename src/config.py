"""YAML設定の読み込みと検証。"""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class AppSection:
    title: str
    max_upload_mb: int


@dataclass(frozen=True, slots=True)
class OllamaSection:
    base_url: str
    embedding_model: str
    generation_model: str
    timeout_seconds: float
    embedding_batch_size: int


@dataclass(frozen=True, slots=True)
class ChunkingSection:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True, slots=True)
class RetrievalSection:
    top_k: int
    distance_threshold: float


@dataclass(frozen=True, slots=True)
class StorageSection:
    documents_dir: Path
    chroma_dir: Path
    database_path: Path
    log_dir: Path
    interaction_log_path: Path
    collection_name: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    project_root: Path
    app: AppSection
    ollama: OllamaSection
    chunking: ChunkingSection
    retrieval: RetrievalSection
    storage: StorageSection

    def ensure_directories(self) -> None:
        self.storage.documents_dir.mkdir(parents=True, exist_ok=True)
        self.storage.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.storage.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage.log_dir.mkdir(parents=True, exist_ok=True)
        self.storage.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)


def _required(mapping: dict, key: str, section: str):
    if key not in mapping:
        raise ConfigurationError(
            "設定ファイルに不足があります。",
            f"Missing key: {section}.{key}",
        )
    return mapping[key]


def _local_ollama_url(value: str) -> str:
    parsed = urlparse(value.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigurationError(
            "Ollamaの接続先設定が正しくありません。",
            f"Invalid Ollama URL: {value!r}",
        )
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ConfigurationError(
            "Ollamaの接続先はこのPC内に限定してください。",
            f"Non-loopback Ollama host rejected: {hostname}",
        )
    return value.rstrip("/")


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    path = Path(config_path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            "設定ファイルを読み込めませんでした。",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        app_raw = _required(raw, "app", "root")
        ollama_raw = _required(raw, "ollama", "root")
        chunk_raw = _required(raw, "chunking", "root")
        retrieval_raw = _required(raw, "retrieval", "root")
        storage_raw = _required(raw, "storage", "root")

        chunk_size = int(_required(chunk_raw, "chunk_size", "chunking"))
        chunk_overlap = int(_required(chunk_raw, "chunk_overlap", "chunking"))
        if chunk_size < 100 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_size must be >= 100 and overlap must be smaller")

        top_k = int(_required(retrieval_raw, "top_k", "retrieval"))
        threshold = float(
            _required(retrieval_raw, "distance_threshold", "retrieval")
        )
        if top_k < 1 or not 0 <= threshold <= 2:
            raise ValueError("retrieval values are outside their supported range")

        project_root = path.parent

        def local_path(key: str) -> Path:
            configured = Path(str(_required(storage_raw, key, "storage")))
            return (
                configured
                if configured.is_absolute()
                else (project_root / configured).resolve()
            )

        config = AppConfig(
            project_root=project_root,
            app=AppSection(
                title=str(_required(app_raw, "title", "app")),
                max_upload_mb=int(_required(app_raw, "max_upload_mb", "app")),
            ),
            ollama=OllamaSection(
                base_url=_local_ollama_url(
                    str(_required(ollama_raw, "base_url", "ollama"))
                ),
                embedding_model=str(
                    _required(ollama_raw, "embedding_model", "ollama")
                ),
                generation_model=str(
                    _required(ollama_raw, "generation_model", "ollama")
                ),
                timeout_seconds=float(
                    _required(ollama_raw, "timeout_seconds", "ollama")
                ),
                embedding_batch_size=int(
                    _required(ollama_raw, "embedding_batch_size", "ollama")
                ),
            ),
            chunking=ChunkingSection(chunk_size, chunk_overlap),
            retrieval=RetrievalSection(top_k, threshold),
            storage=StorageSection(
                documents_dir=local_path("documents_dir"),
                chroma_dir=local_path("chroma_dir"),
                database_path=local_path("database_path"),
                log_dir=local_path("log_dir"),
                interaction_log_path=local_path("interaction_log_path"),
                collection_name=str(
                    _required(storage_raw, "collection_name", "storage")
                ),
            ),
        )
    except ConfigurationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            "設定値に誤りがあります。管理者に確認してください。",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    if config.app.max_upload_mb < 1 or config.ollama.embedding_batch_size < 1:
        raise ConfigurationError(
            "設定値に誤りがあります。管理者に確認してください。",
            "Upload size and embedding batch size must be positive.",
        )
    config.ensure_directories()
    return config
