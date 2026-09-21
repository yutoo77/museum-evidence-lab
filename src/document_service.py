"""資料の重複判定、読込、分割、ベクトル化、永続化を統括する。"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.chunking import chunk_document
from src.config import AppConfig
from src.database import DocumentRepository
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingProvider
from src.exceptions import DocumentRegistrationError, RAGApplicationError
from src.models import (
    DocumentRecord,
    RegistrationResult,
    RegistrationStatus,
)
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


def logical_document_name(filename: str) -> str:
    return unicodedata.normalize("NFKC", Path(filename).name).casefold()


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ".bin"


class DocumentRegistrationService:
    def __init__(
        self,
        config: AppConfig,
        repository: DocumentRepository,
        loader: DocumentLoader,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.repository = repository
        self.loader = loader
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def register_bytes(self, filename: str, data: bytes) -> RegistrationResult:
        source_name = Path(filename).name.strip()
        if not source_name:
            raise DocumentRegistrationError(
                "資料名を確認できませんでした。",
                "Uploaded filename is empty.",
            )
        maximum = self.config.app.max_upload_mb * 1024 * 1024
        if len(data) > maximum:
            raise DocumentRegistrationError(
                f"ファイルサイズが上限（{self.config.app.max_upload_mb}MB）を超えています。",
                f"Upload size {len(data)} exceeds {maximum} bytes.",
            )

        content_hash = hashlib.sha256(data).hexdigest()
        duplicate = self.repository.find_by_content_hash(content_hash)
        if duplicate is not None:
            return RegistrationResult(
                status=RegistrationStatus.ALREADY_REGISTERED,
                record=duplicate,
                message=f"同じ内容の資料「{duplicate.source_name}」は登録済みです。",
            )

        logical_name = logical_document_name(source_name)
        previous = self.repository.find_by_logical_name(logical_name)
        document_id = previous.document_id if previous else str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        registered_at = datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        vector_written = False
        source_directory = self.config.storage.documents_dir / document_id / version_id
        original_path = source_directory / f"original{_safe_suffix(source_name)}"

        try:
            document = self.loader.load_bytes(source_name, data)
            chunks = chunk_document(
                document,
                self.config.chunking.chunk_size,
                self.config.chunking.chunk_overlap,
            )
            embeddings = self.embedding_provider.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_store.upsert_version(
                document_id=document_id,
                version_id=version_id,
                source_name=source_name,
                file_type=document.file_type,
                content_hash=content_hash,
                registered_at=registered_at,
                chunks=chunks,
                embeddings=embeddings,
            )
            vector_written = True

            source_directory.mkdir(parents=True, exist_ok=False)
            original_path.write_bytes(data)
            record = DocumentRecord(
                document_id=document_id,
                version_id=version_id,
                logical_name=logical_name,
                source_name=source_name,
                file_type=document.file_type,
                content_hash=content_hash,
                page_count=document.page_count,
                chunk_count=len(chunks),
                registered_at=registered_at,
                status="registered",
                original_path=original_path,
                warning=document.warning,
            )
            self.repository.upsert(record)
        except RAGApplicationError:
            if vector_written:
                self._cleanup_vector(version_id)
            self._cleanup_source(source_directory)
            raise
        except Exception as exc:
            if vector_written:
                self._cleanup_vector(version_id)
            self._cleanup_source(source_directory)
            raise DocumentRegistrationError(
                "資料の登録中に予期しないエラーが発生しました。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

        if previous:
            try:
                self.vector_store.delete_version(previous.version_id)
            except RAGApplicationError as exc:
                logger.warning(
                    "Failed to remove old vector version %s: %s",
                    previous.version_id,
                    exc.technical_detail,
                )
            if previous.original_path:
                self._cleanup_source(previous.original_path.parent)

        status = RegistrationStatus.REPLACED if previous else RegistrationStatus.REGISTERED
        action = "再登録" if previous else "登録"
        return RegistrationResult(
            status=status,
            record=record,
            message=f"「{source_name}」を検索用資料として{action}しました。",
        )

    def _cleanup_vector(self, version_id: str) -> None:
        try:
            self.vector_store.delete_version(version_id)
        except Exception:
            logger.exception("Failed to clean up vector version %s", version_id)

    @staticmethod
    def _cleanup_source(directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        parent = directory.parent
        try:
            parent.rmdir()
        except OSError:
            pass
