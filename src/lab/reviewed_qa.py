"""Staff-confirmed explanations and their original-document provenance."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class QASourceRef:
    source_name: str
    content_hash: str
    evidence_id: str
    quote: str


@dataclass(frozen=True)
class ReviewedQA:
    id: int
    question: str
    original_generated_answer: str
    approved_answer: str
    created_at: str
    updated_at: str
    status: str
    source_refs: tuple[QASourceRef, ...]
    audience: str = "general"
    detail: str = "standard"
    distance: float = 1.0
    lexical_score: float = 0.0

    @property
    def source_evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ref.evidence_id for ref in self.source_refs))


def ensure_reviewed_qa_schema(db: sqlite3.Connection) -> None:
    """Add tables without changing the existing document-index fingerprint."""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS reviewed_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            original_generated_answer TEXT NOT NULL,
            approved_answer TEXT NOT NULL,
            audience TEXT NOT NULL,
            detail TEXT NOT NULL,
            vector TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('confirmed', 'needs_review', 'archived'))
        );
        CREATE TABLE IF NOT EXISTS reviewed_qa_sources (
            qa_id INTEGER NOT NULL,
            logical_source TEXT NOT NULL,
            source_name TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            quote TEXT NOT NULL,
            PRIMARY KEY (qa_id, evidence_id, quote),
            FOREIGN KEY (qa_id) REFERENCES reviewed_qa(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS reviewed_qa_lookup
            ON reviewed_qa(status, audience, detail);
        CREATE INDEX IF NOT EXISTS reviewed_qa_source_lookup
            ON reviewed_qa_sources(logical_source, qa_id);
        """
    )


def load_reviewed_qa(
    db: sqlite3.Connection,
    *,
    status: str | None = None,
    audience: str | None = None,
    detail: str | None = None,
) -> list[ReviewedQA]:
    conditions: list[str] = []
    arguments: list[str] = []
    for column, value in (
        ("status", status),
        ("audience", audience),
        ("detail", detail),
    ):
        if value is not None:
            conditions.append(f"{column} = ?")
            arguments.append(value)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    records = db.execute(
        "SELECT * FROM reviewed_qa" + where + " ORDER BY id DESC", arguments
    ).fetchall()
    if not records:
        return []
    identifiers = [row["id"] for row in records]
    placeholders = ", ".join("?" for _ in identifiers)
    references: dict[int, list[QASourceRef]] = {
        identifier: [] for identifier in identifiers
    }
    for row in db.execute(
        "SELECT qa_id, source_name, content_hash, evidence_id, quote "
        "FROM reviewed_qa_sources WHERE qa_id IN (" + placeholders + ") "
        "ORDER BY qa_id, evidence_id, quote",
        identifiers,
    ):
        references[row["qa_id"]].append(
            QASourceRef(
                row["source_name"],
                row["content_hash"],
                row["evidence_id"],
                row["quote"],
            )
        )
    return [
        ReviewedQA(
            id=row["id"],
            question=row["question"],
            original_generated_answer=row["original_generated_answer"],
            approved_answer=row["approved_answer"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            source_refs=tuple(references[row["id"]]),
            audience=row["audience"],
            detail=row["detail"],
        )
        for row in records
    ]
