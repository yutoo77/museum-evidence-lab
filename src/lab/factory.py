"""Assemble isolated application services with versioned local indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.lab.audit import AuditLog
from src.lab.client import InstrumentedOllamaClient, SafeEmbeddingClient
from src.lab.engine import LabEngine
from src.lab.faq import ApprovedFAQ
from src.lab.index import HybridIndex
from src.lab.settings import load_lab_settings


@dataclass
class LabBundle:
    settings: object
    index: HybridIndex
    client: InstrumentedOllamaClient
    embedding: SafeEmbeddingClient
    engine: LabEngine
    faq: ApprovedFAQ

    def close(self):
        self.client.close()
        self.embedding.close()


def build_lab(config_path="lab_config.yaml", *, data_dir=None, generation_model=None):
    settings = load_lab_settings(config_path)
    root = Path(data_dir) if data_dir is not None else settings.storage.root
    root.mkdir(parents=True, exist_ok=True)
    ollama = settings.ollama
    embedding = SafeEmbeddingClient(
        ollama.base_url,
        ollama.embedding_model,
        ollama.allowed_models,
        timeout_seconds=ollama.timeout_seconds,
    )
    client = InstrumentedOllamaClient(
        ollama.base_url,
        generation_model or ollama.generation_model,
        ollama.allowed_models,
        timeout_seconds=ollama.timeout_seconds,
        context_length=ollama.context_length,
    )
    try:
        index = HybridIndex(
            root / "index.sqlite3",
            embedding,
            embedding_digest=embedding.model_digest,
            chunk_size=settings.retrieval.chunk_size,
            chunk_overlap=settings.retrieval.overlap,
        )
        faq = ApprovedFAQ(root / "approved_faq.sqlite3")
        engine = LabEngine(
            index,
            client,
            faq=faq,
            audit=AuditLog(
                root / "audit",
                enabled=settings.logging.enabled,
                retention_days=settings.logging.retention_days,
            ),
            max_evidence=settings.retrieval.max_evidence,
            max_context_chars=settings.retrieval.max_context_chars,
            threshold=settings.retrieval.threshold,
            budget_seconds=ollama.timeout_seconds,
        )
        return LabBundle(settings, index, client, embedding, engine, faq)
    except Exception:
        embedding.close()
        client.close()
        raise
