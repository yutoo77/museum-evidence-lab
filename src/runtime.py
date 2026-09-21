"""画面から利用するサービス群の組み立て。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import AppConfig, load_config
from src.database import DocumentRepository
from src.document_loader import DocumentLoader
from src.document_service import DocumentRegistrationService
from src.embeddings import OllamaEmbeddingClient
from src.generation import OllamaChatClient
from src.interaction_logs import InteractionLogStore
from src.logs import configure_logging
from src.qa_service import QuestionAnsweringService
from src.retrieval import RetrievalService
from src.vector_store import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    config: AppConfig
    repository: DocumentRepository
    embedding_client: OllamaEmbeddingClient
    generation_client: OllamaChatClient
    vector_store: ChromaVectorStore
    registration_service: DocumentRegistrationService
    interaction_log: InteractionLogStore
    qa_service: QuestionAnsweringService


def build_services(config_path: str | Path = "config.yaml") -> ApplicationServices:
    config = load_config(config_path)

    return build_services_from_config(config)


def build_services_from_config(config: AppConfig) -> ApplicationServices:
    """検証済み設定からサービスを構築する。

    通常アプリは``build_services``を使い、Benchmarkは実行ごとに隔離した
    Storage設定を渡す。外部通信先の検証は``load_config``済み設定を前提とする。
    """

    configure_logging(config.storage.log_dir)
    repository = DocumentRepository(config.storage.database_path)
    embedding_client = OllamaEmbeddingClient(
        base_url=config.ollama.base_url,
        model_name=config.ollama.embedding_model,
        timeout_seconds=config.ollama.timeout_seconds,
        batch_size=config.ollama.embedding_batch_size,
    )
    generation_client = OllamaChatClient(
        base_url=config.ollama.base_url,
        model_name=config.ollama.generation_model,
        timeout_seconds=config.ollama.timeout_seconds,
    )
    vector_store = ChromaVectorStore(
        config.storage.chroma_dir,
        config.storage.collection_name,
    )
    registration_service = DocumentRegistrationService(
        config=config,
        repository=repository,
        loader=DocumentLoader(),
        embedding_provider=embedding_client,
        vector_store=vector_store,
    )
    interaction_log = InteractionLogStore(config.storage.interaction_log_path)
    retrieval_service = RetrievalService(
        repository=repository,
        embedding_provider=embedding_client,
        vector_store=vector_store,
    )
    qa_service = QuestionAnsweringService(
        retrieval_service=retrieval_service,
        generation_provider=generation_client,
        log_store=interaction_log,
        embedding_model=config.ollama.embedding_model,
    )
    return ApplicationServices(
        config=config,
        repository=repository,
        embedding_client=embedding_client,
        generation_client=generation_client,
        vector_store=vector_store,
        registration_service=registration_service,
        interaction_log=interaction_log,
        qa_service=qa_service,
    )
