"""資料台帳を管理するSQLiteリポジトリ。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.models import DocumentRecord


class DocumentRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    logical_name TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    registered_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    original_path TEXT,
                    warning TEXT NOT NULL DEFAULT '',
                    error_detail TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)"
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> DocumentRecord | None:
        if row is None:
            return None
        return DocumentRecord(
            document_id=row["document_id"],
            version_id=row["version_id"],
            logical_name=row["logical_name"],
            source_name=row["source_name"],
            file_type=row["file_type"],
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            registered_at=row["registered_at"],
            status=row["status"],
            original_path=Path(row["original_path"]) if row["original_path"] else None,
            warning=row["warning"],
            error_detail=row["error_detail"],
        )

    def find_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return self._row_to_record(row)

    def find_by_logical_name(self, logical_name: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE logical_name = ?",
                (logical_name,),
            ).fetchone()
        return self._row_to_record(row)

    def upsert(self, record: DocumentRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, version_id, logical_name, source_name, file_type,
                    content_hash, page_count, chunk_count, registered_at, status,
                    original_path, warning, error_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    version_id = excluded.version_id,
                    logical_name = excluded.logical_name,
                    source_name = excluded.source_name,
                    file_type = excluded.file_type,
                    content_hash = excluded.content_hash,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    registered_at = excluded.registered_at,
                    status = excluded.status,
                    original_path = excluded.original_path,
                    warning = excluded.warning,
                    error_detail = excluded.error_detail
                """,
                (
                    record.document_id,
                    record.version_id,
                    record.logical_name,
                    record.source_name,
                    record.file_type,
                    record.content_hash,
                    record.page_count,
                    record.chunk_count,
                    record.registered_at,
                    record.status,
                    str(record.original_path) if record.original_path else None,
                    record.warning,
                    record.error_detail,
                ),
            )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY registered_at DESC, source_name"
            ).fetchall()
        return [record for row in rows if (record := self._row_to_record(row))]

    def active_version_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT version_id FROM documents WHERE status = 'registered'"
            ).fetchall()
        return [str(row["version_id"]) for row in rows]
