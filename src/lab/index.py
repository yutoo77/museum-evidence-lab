"""Local SQLite evidence index with bounded, inspectable hybrid retrieval.

The original files and vectors stay in this database. Lexical BM25 scores are
ranking signals, never probabilities. ``dense`` deliberately preserves the old
unscreened top-five retrieval for a reproducible reference comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.chunking import chunk_document
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingProvider
from src.lab.contracts import Evidence, LabAnswer, SearchResult
from src.lab.explanation import fact_tokens
from src.lab.grounding import INJECTION, REMOTE
from src.lab.reviewed_qa import (
    QASourceRef,
    ReviewedQA,
    ensure_reviewed_qa_schema,
    load_reviewed_qa,
)

MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_SCHEMA_VERSION = 1
_DOCUMENT_METADATA = (
    "source_name, content_hash, page_count, chunk_count, approved, effective_date"
)
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")
_WORDS = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_BOILERPLATE = re.compile(
    r"教えてください|教えて|について|何ですか|なんですか|なのですか|ですか|でしょうか|とは|ください"
)
_STOP_WORDS = {"a", "an", "the", "is", "are", "what", "how", "of", "to", "in", "and"}
_STOP_GRAMS = {"です", "ます", "する", "して", "とは", "は何", "何か", "もの", "こと"}
_QA_AUDIENCES = {"general", "child", "staff"}
_QA_DETAILS = {"short", "standard"}
# Reviewed text can steer a later explanation, so require a closer semantic match.
_QA_DISTANCE_CAP = 0.45


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class IndexMismatchError(ValueError):
    """An existing index needs to be rebuilt with the selected configuration."""


class InvalidEmbeddingError(ValueError):
    """A provider returned unusable or incompatible embeddings."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _terms(text: str) -> Counter[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _BOILERPLATE.sub(" ", normalized)
    tokens = [
        "w:" + word for word in _WORDS.findall(normalized) if word not in _STOP_WORDS
    ]
    for run in _JAPANESE.findall(normalized):
        tokens.extend(
            "j:" + run[i : i + 2]
            for i in range(len(run) - 1)
            if run[i : i + 2] not in _STOP_GRAMS
        )
    return Counter(tokens)


def _unit_vector(values: Any, expected_dimension: int | None = None) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidEmbeddingError(
            "検索用の数値が不正です。資料を再登録してください。"
        ) from exc
    if not vector or any(not math.isfinite(value) for value in vector):
        raise InvalidEmbeddingError(
            "検索用の数値に空データ・非有限値が含まれています。"
        )
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise IndexMismatchError(
            "検索モデルの次元が変更されています。新しい索引を作成してください。"
        )
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm <= 0:
        raise InvalidEmbeddingError("検索用の数値がゼロまたは不正です。")
    return [value / norm for value in vector]


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("検索処理の制限時間を超えました。")
    return remaining


class HybridIndex:
    """An isolated, content-versioned evidence store; no Chroma server required."""

    def __init__(
        self,
        db_path: str | Path,
        embedding_provider: EmbeddingProvider,
        *,
        embedding_digest: str,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be smaller than positive chunk_size")
        if not embedding_digest.strip():
            raise ValueError("embedding_digest must identify the actual model weights")
        self.db_path = Path(db_path)
        self.provider = embedding_provider
        self.embedding_digest = embedding_digest
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    logical_source TEXT PRIMARY KEY, source_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL, original BLOB NOT NULL,
                    page_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    approved INTEGER NOT NULL, effective_date TEXT
                );
                CREATE TABLE IF NOT EXISTS passages (
                    evidence_id TEXT PRIMARY KEY, logical_source TEXT NOT NULL,
                    text TEXT NOT NULL, page_number INTEGER, vector TEXT NOT NULL,
                    FOREIGN KEY(logical_source) REFERENCES documents(logical_source) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS documents_content_hash ON documents(content_hash);
                """
            )
            ensure_reviewed_qa_schema(db)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(str(self.db_path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _configuration(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "model_name": self.provider.model_name,
            "model_digest": self.embedding_digest,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _check_returned_digest(self) -> None:
        """Reject a tag whose weights changed after this index was opened."""
        metrics = getattr(self.provider, "last_metrics", None)
        returned = metrics.get("digest") if isinstance(metrics, dict) else None
        if returned is not None and returned != self.embedding_digest:
            raise IndexMismatchError(
                "検索モデルの内容が変更されています。新しい索引を作成してください。"
            )

    def _check_fingerprint(self, db: sqlite3.Connection) -> dict[str, Any] | None:
        record = db.execute(
            "SELECT value FROM metadata WHERE key = 'fingerprint'"
        ).fetchone()
        if record is None:
            return None
        stored = json.loads(record["value"])
        if any(
            stored.get(key) != value for key, value in self._configuration().items()
        ):
            raise IndexMismatchError(
                "検索モデル・モデルの版・資料の分割設定が索引と一致しません。別の索引を作成してください。"
            )
        return stored

    @staticmethod
    def _source_name(filename: str) -> str:
        source = unicodedata.normalize(
            "NFKC", filename.replace("\\", "/").rsplit("/", 1)[-1]
        ).strip()
        if (
            not source
            or source in {".", ".."}
            or any(ord(char) < 32 for char in source)
        ):
            raise ValueError("資料名が不正です。")
        return source

    @staticmethod
    def _date(value: str | date | None) -> str | None:
        if value is None:
            return None
        return date.fromisoformat(str(value)).isoformat()

    @staticmethod
    def _metadata(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "source_name": row["source_name"],
            "content_hash": row["content_hash"],
            "page_count": row["page_count"],
            "chunk_count": row["chunk_count"],
            "approved": bool(row["approved"]),
            "effective_date": row["effective_date"],
        }

    def register_bytes(
        self,
        filename: str,
        data: bytes,
        *,
        approved: bool = True,
        effective_date: str | date | None = None,
    ) -> dict[str, Any]:
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError("資料は50 MB以内にしてください。")
        if not isinstance(approved, bool):
            raise ValueError("approved must be a boolean")
        source_name = self._source_name(filename)
        logical_source = source_name.casefold()
        effective = self._date(effective_date)
        digest = hashlib.sha256(data).hexdigest()
        with self._connect() as db:
            fingerprint = self._check_fingerprint(db)
            old = db.execute(
                f"SELECT {_DOCUMENT_METADATA} FROM documents WHERE logical_source = ?",
                (logical_source,),
            ).fetchone()
            if (
                old is not None
                and old["content_hash"] == digest
                and old["source_name"] == source_name
                and bool(old["approved"]) == approved
                and old["effective_date"] == effective
            ):
                return self._metadata(old)

        document = DocumentLoader().load_bytes(source_name, data)
        chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)
        raw_vectors = self.provider.embed_texts([chunk.text for chunk in chunks])
        self._check_returned_digest()
        if len(raw_vectors) != len(chunks):
            raise InvalidEmbeddingError("資料数と検索用の数値の件数が一致しません。")
        dimension = fingerprint["dimension"] if fingerprint else None
        vectors: list[list[float]] = []
        for raw in raw_vectors:
            vector = _unit_vector(raw, dimension)
            dimension = len(vector)
            vectors.append(vector)
        new_fingerprint = {**self._configuration(), "dimension": dimension}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._check_fingerprint(db)
            if current is not None and current != new_fingerprint:
                raise IndexMismatchError("検索モデルの次元が索引と一致しません。")
            current_document = db.execute(
                "SELECT 1 FROM documents WHERE logical_source = ?", (logical_source,)
            ).fetchone()
            if current_document is not None:
                db.execute(
                    """UPDATE reviewed_qa SET status = 'needs_review', updated_at = ?
                    WHERE status = 'confirmed' AND id IN (
                        SELECT qa_id FROM reviewed_qa_sources WHERE logical_source = ?
                    )""",
                    (_utc_now(), logical_source),
                )
            db.execute(
                "DELETE FROM documents WHERE logical_source = ?", (logical_source,)
            )
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    logical_source,
                    source_name,
                    digest,
                    data,
                    document.page_count,
                    len(chunks),
                    approved,
                    effective,
                ),
            )
            for chunk, vector in zip(chunks, vectors, strict=True):
                identity = _canonical(
                    [logical_source, digest, chunk.chunk_number, chunk.text]
                )
                evidence_id = (
                    "E" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                )
                db.execute(
                    "INSERT INTO passages VALUES (?, ?, ?, ?, ?)",
                    (
                        evidence_id,
                        logical_source,
                        chunk.text,
                        chunk.page_number,
                        _canonical(vector),
                    ),
                )
            db.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('fingerprint', ?)",
                (_canonical(new_fingerprint),),
            )
            row = db.execute(
                f"SELECT {_DOCUMENT_METADATA} FROM documents WHERE logical_source = ?",
                (logical_source,),
            ).fetchone()
            return self._metadata(row)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [
                self._metadata(row)
                for row in db.execute(
                    f"SELECT {_DOCUMENT_METADATA} FROM documents ORDER BY logical_source"
                )
            ]

    def get_document(self, content_hash: str) -> dict[str, Any] | None:
        """Return original bytes for a current document; never construct file paths."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY logical_source LIMIT 1",
                (content_hash,),
            ).fetchone()
            if row is None:
                return None
            mime_type = (
                "application/pdf"
                if row["source_name"].lower().endswith(".pdf")
                else "text/plain"
            )
            return {
                **self._metadata(row),
                "data": bytes(row["original"]),
                "mime_type": mime_type,
            }

    def _snapshot(self) -> tuple[list[tuple[Evidence, list[float]]], str, int | None]:
        with self._connect() as db:
            db.execute("BEGIN")
            fingerprint = self._check_fingerprint(db)
            rows = db.execute(
                """SELECT p.*, d.source_name, d.content_hash FROM passages p
                JOIN documents d ON d.logical_source = p.logical_source
                WHERE d.approved = 1 AND (d.effective_date IS NULL OR d.effective_date <= ?)
                ORDER BY p.evidence_id""",
                (date.today().isoformat(),),
            ).fetchall()
            active_ids = [row["evidence_id"] for row in rows]
            revision = self._revision(db, fingerprint, active_ids)
            dimension = fingerprint["dimension"] if fingerprint else None
            passages = [
                (
                    Evidence(
                        evidence_id=row["evidence_id"],
                        text=row["text"],
                        source_name=row["source_name"],
                        page_number=row["page_number"],
                        content_hash=row["content_hash"],
                        version_id=row["content_hash"],
                    ),
                    _unit_vector(json.loads(row["vector"]), dimension),
                )
                for row in rows
            ]
            return passages, revision, dimension

    @property
    def revision(self) -> str:
        """Check cache identity without decoding document bodies or vectors."""
        with self._connect() as db:
            db.execute("BEGIN")
            fingerprint = self._check_fingerprint(db)
            active_ids = [
                row[0]
                for row in db.execute(
                    """SELECT p.evidence_id FROM passages p
                JOIN documents d ON d.logical_source = p.logical_source
                WHERE d.approved = 1 AND (d.effective_date IS NULL OR d.effective_date <= ?)
                ORDER BY p.evidence_id""",
                    (date.today().isoformat(),),
                )
            ]
            return self._revision(db, fingerprint, active_ids)

    @staticmethod
    def _revision(
        db: sqlite3.Connection,
        fingerprint: dict[str, Any] | None,
        active_ids: list[str],
    ) -> str:
        docs = [
            tuple(row)
            for row in db.execute(
                """SELECT logical_source, source_name, content_hash, approved, effective_date
            FROM documents ORDER BY logical_source"""
            )
        ]
        qa_states = [
            tuple(row)
            for row in db.execute(
                "SELECT id, status, updated_at FROM reviewed_qa ORDER BY id"
            )
        ]
        return hashlib.sha256(
            _canonical([fingerprint, docs, active_ids, qa_states]).encode("utf-8")
        ).hexdigest()

    def export_passages(self) -> tuple[Evidence, ...]:
        return tuple(evidence for evidence, _ in self._snapshot()[0])

    def save_reviewed_qa(
        self,
        question: str,
        answer: LabAnswer,
        approved_answer: str,
        expected_revision: str,
        *,
        audience: str = "general",
        detail: str = "standard",
    ) -> ReviewedQA:
        """Save a staff-confirmed explanation only against its current sources."""
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > 1500
            or not isinstance(approved_answer, str)
            or not approved_answer.strip()
            or len(approved_answer) > 803
            or not isinstance(expected_revision, str)
            or not expected_revision
            or not isinstance(audience, str)
            or not isinstance(detail, str)
            or audience not in _QA_AUDIENCES
            or detail not in _QA_DETAILS
        ):
            raise ValueError("質問・確認済み回答・検索状態を確認してください。")
        if (
            not isinstance(answer, LabAnswer)
            or answer.status != "answered"
            or answer.route != "explain"
        ):
            raise ValueError("根拠のある説明案だけを確認済みQ&Aとして保存できます。")
        question = question.strip()
        approved_answer = approved_answer.strip()
        if (
            INJECTION.search(question)
            or REMOTE.search(question)
            or INJECTION.search(approved_answer)
            or REMOTE.search(approved_answer)
        ):
            raise ValueError("この内容は確認済みQ&Aとして保存できません。")
        original_answer = answer.text.strip()
        if not original_answer or len(original_answer) > 1500:
            raise ValueError("元の説明案を確認できません。")
        by_id = {passage.evidence_id: passage for passage in answer.evidence}
        source_refs: dict[tuple[str, str], QASourceRef] = {}
        for claim in answer.claims:
            if not claim.references:
                raise ValueError("根拠のない説明案は保存できません。")
            for reference in claim.references:
                passage = by_id.get(reference.evidence_id)
                if (
                    passage is None
                    or not reference.quote
                    or reference.quote not in passage.text
                    or not passage.content_hash
                ):
                    raise ValueError("引用元を確認できない説明案は保存できません。")
                source_refs[(reference.evidence_id, reference.quote)] = QASourceRef(
                    passage.source_name,
                    passage.content_hash,
                    reference.evidence_id,
                    reference.quote,
                )
        if not source_refs:
            raise ValueError("根拠のない説明案は保存できません。")
        supported_tokens = set().union(
            *(fact_tokens(ref.quote) for ref in source_refs.values())
        )
        if not fact_tokens(approved_answer) <= supported_tokens:
            raise ValueError(
                "修正後の回答に、引用元で確認できない数値や識別子があります。"
            )
        raw_vectors = self.provider.embed_texts([question])
        self._check_returned_digest()
        if len(raw_vectors) != 1:
            raise InvalidEmbeddingError("確認済みQ&Aの検索用の数値が不正です。")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            fingerprint = self._check_fingerprint(db)
            active_ids = [
                row[0]
                for row in db.execute(
                    """SELECT p.evidence_id FROM passages p
                    JOIN documents d ON d.logical_source = p.logical_source
                    WHERE d.approved = 1 AND (d.effective_date IS NULL OR d.effective_date <= ?)
                    ORDER BY p.evidence_id""",
                    (date.today().isoformat(),),
                )
            ]
            if self._revision(db, fingerprint, active_ids) != expected_revision:
                raise ValueError(
                    "資料や確認済みQ&Aが更新されました。もう一度質問してください。"
                )
            if fingerprint is None:
                raise ValueError("根拠資料を確認できません。")
            vector = _unit_vector(raw_vectors[0], fingerprint["dimension"])
            for reference in source_refs.values():
                row = db.execute(
                    """SELECT p.text, d.logical_source, d.source_name, d.content_hash
                    FROM passages p JOIN documents d ON d.logical_source = p.logical_source
                    WHERE p.evidence_id = ? AND d.approved = 1
                      AND (d.effective_date IS NULL OR d.effective_date <= ?)""",
                    (reference.evidence_id, date.today().isoformat()),
                ).fetchone()
                if (
                    row is None
                    or row["source_name"] != reference.source_name
                    or row["content_hash"] != reference.content_hash
                    or reference.quote not in row["text"]
                ):
                    raise ValueError(
                        "参照資料が更新されました。もう一度質問してください。"
                    )
            now = _utc_now()
            cursor = db.execute(
                """INSERT INTO reviewed_qa
                (question, original_generated_answer, approved_answer, audience,
                 detail, vector, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')""",
                (
                    question,
                    original_answer,
                    approved_answer,
                    audience,
                    detail,
                    _canonical(vector),
                    now,
                    now,
                ),
            )
            identifier = int(cursor.lastrowid)
            for reference in source_refs.values():
                db.execute(
                    """INSERT INTO reviewed_qa_sources
                    (qa_id, logical_source, source_name, content_hash, evidence_id, quote)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        identifier,
                        reference.source_name.casefold(),
                        reference.source_name,
                        reference.content_hash,
                        reference.evidence_id,
                        reference.quote,
                    ),
                )
        ordered_refs = tuple(
            sorted(source_refs.values(), key=lambda ref: (ref.evidence_id, ref.quote))
        )
        return ReviewedQA(
            identifier,
            question,
            original_answer,
            approved_answer,
            now,
            now,
            "confirmed",
            ordered_refs,
            audience,
            detail,
        )

    def list_reviewed_qa(self) -> list[ReviewedQA]:
        with self._connect() as db:
            return load_reviewed_qa(db)

    def archive_reviewed_qa(self, identifier: int) -> None:
        if type(identifier) is not int or identifier < 1:
            raise ValueError("確認済みQ&Aの番号を確認してください。")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM reviewed_qa WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise ValueError("確認済みQ&Aが見つかりません。")
            if row["status"] != "archived":
                db.execute(
                    "UPDATE reviewed_qa SET status = 'archived', updated_at = ? WHERE id = ?",
                    (_utc_now(), identifier),
                )

    @staticmethod
    def _lexical_scores(
        question: str, evidence: list[Evidence]
    ) -> tuple[list[float], list[bool]]:
        query = _terms(question)
        documents = [_terms(item.text) for item in evidence]
        if not query or not documents:
            return [0.0] * len(documents), [False] * len(documents)
        frequency = Counter(term for document in documents for term in document)
        count = len(documents)
        idf = {
            term: math.log(
                1 + (count - frequency[term] + 0.5) / (frequency[term] + 0.5)
            )
            for term in query
        }
        average_length = (
            sum(sum(document.values()) for document in documents) / count or 1.0
        )
        scores: list[float] = []
        strong: list[bool] = []
        for document in documents:
            length = sum(document.values())
            score = sum(
                idf[term]
                * document[term]
                * 2.2
                / (document[term] + 1.2 * (0.25 + 0.75 * length / average_length))
                for term in query
                if document[term]
            )
            matched = set(query) & set(document)
            coverage = sum(idf[term] for term in matched) / sum(idf.values())
            scores.append(score)
            strong.append(coverage >= 0.5 and len(matched) >= min(2, len(query)))
        return scores, strong

    def _rank_reviewed_qa(
        self,
        question: str,
        active_passages: list[tuple[Evidence, list[float]]],
        query_vector: list[float] | None,
        dimension: int | None,
        *,
        audience: str,
        detail: str,
        threshold: float,
        deadline: float | None,
    ) -> tuple[ReviewedQA, ...]:
        """Use the document search signals; never treat Q&A as original evidence."""
        with self._connect() as db:
            records = load_reviewed_qa(
                db, status="confirmed", audience=audience, detail=detail
            )
            if not records:
                return ()
            vectors = {
                row["id"]: row["vector"]
                for row in db.execute(
                    "SELECT id, vector FROM reviewed_qa WHERE status = 'confirmed' "
                    "AND audience = ? AND detail = ?",
                    (audience, detail),
                )
            }
        _remaining(deadline)
        by_id = {passage.evidence_id: passage for passage, _ in active_passages}
        eligible: list[ReviewedQA] = []
        qa_vectors: list[list[float]] = []
        for record in records:
            if (
                not record.source_refs
                or INJECTION.search(record.approved_answer)
                or REMOTE.search(record.approved_answer)
                or any(
                    (passage := by_id.get(ref.evidence_id)) is None
                    or passage.source_name != ref.source_name
                    or passage.content_hash != ref.content_hash
                    or ref.quote not in passage.text
                    for ref in record.source_refs
                )
            ):
                continue
            try:
                vector = _unit_vector(json.loads(vectors[record.id]), dimension)
            except (
                KeyError,
                ValueError,
                TypeError,
                InvalidEmbeddingError,
                IndexMismatchError,
            ):
                continue
            eligible.append(record)
            qa_vectors.append(vector)
        if not eligible:
            return ()
        lexical, strong_lexical = self._lexical_scores(
            question,
            [
                Evidence(f"Q{record.id}", record.question, "", None, "")
                for record in eligible
            ],
        )
        if query_vector is None:
            distances = [1.0] * len(eligible)
        else:
            distances = [
                1.0
                - max(
                    -1.0,
                    min(
                        1.0,
                        sum(a * b for a, b in zip(query_vector, vector, strict=True)),
                    ),
                )
                for vector in qa_vectors
            ]
        dense_order = sorted(
            range(len(eligible)), key=lambda i: (distances[i], -eligible[i].id)
        )
        lexical_order = sorted(
            range(len(eligible)), key=lambda i: (-lexical[i], -eligible[i].id)
        )
        if query_vector is None:
            order = lexical_order
        else:
            rrf = Counter(
                {i: 1.0 / (60 + rank) for rank, i in enumerate(dense_order, 1)}
            )
            for rank, i in enumerate(lexical_order, 1):
                if lexical[i] > 0:
                    rrf[i] += 1.0 / (60 + rank)
            order = sorted(
                range(len(eligible)),
                key=lambda i: (-rrf[i], distances[i], -eligible[i].id),
            )
        best_distance = min(distances)
        best_lexical = max(lexical)
        selected: list[ReviewedQA] = []
        for i in order:
            dense_relevant = query_vector is not None and distances[i] <= min(
                threshold, _QA_DISTANCE_CAP, best_distance + 0.2
            )
            lexical_relevant = strong_lexical[i] and lexical[i] >= best_lexical * 0.35
            if dense_relevant or lexical_relevant:
                _remaining(deadline)
                selected.append(
                    replace(
                        eligible[i], distance=distances[i], lexical_score=lexical[i]
                    )
                )
                if len(selected) >= 3:
                    break
        return tuple(selected)

    def search(
        self,
        question: str,
        *,
        mode: str = "hybrid",
        max_evidence: int = 3,
        max_context_chars: int = 1800,
        threshold: float = 0.75,
        timeout_seconds: float | None = None,
        include_reviewed_qa: bool = False,
        reviewed_audience: str = "general",
        reviewed_detail: str = "standard",
    ) -> SearchResult:
        started = time.perf_counter()
        if timeout_seconds is not None and (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("検索の制限時間は正の有限秒数で指定してください。")
        deadline = started + timeout_seconds if timeout_seconds is not None else None
        if mode not in {"hybrid", "dense", "lexical"}:
            raise ValueError("mode must be hybrid, dense, or lexical")
        if type(include_reviewed_qa) is not bool or (
            include_reviewed_qa
            and (
                not isinstance(reviewed_audience, str)
                or not isinstance(reviewed_detail, str)
                or reviewed_audience not in _QA_AUDIENCES
                or reviewed_detail not in _QA_DETAILS
            )
        ):
            raise ValueError("確認済みQ&Aの検索条件を確認してください。")
        if max_evidence < 1 or max_context_chars < 1 or not 0 <= threshold <= 2:
            raise ValueError(
                "invalid evidence limit, context budget, or distance threshold"
            )
        passages, revision, dimension = self._snapshot()
        _remaining(deadline)
        if not question.strip() or not passages:
            return SearchResult(
                (), elapsed_seconds=time.perf_counter() - started, revision=revision
            )
        evidence = [item for item, _ in passages]
        embedding_seconds = 0.0
        distances = [1.0] * len(passages)
        query_vector = None
        if mode != "lexical":
            embedding_started = time.perf_counter()
            kwargs = (
                {"timeout_seconds": _remaining(deadline)}
                if deadline is not None
                else {}
            )
            raw = self.provider.embed_texts([question], **kwargs)
            self._check_returned_digest()
            _remaining(deadline)
            if len(raw) != 1:
                raise InvalidEmbeddingError("質問の検索用数値の件数が不正です。")
            query = _unit_vector(raw[0], dimension)
            query_vector = query
            embedding_seconds = time.perf_counter() - embedding_started
            distances = [
                1.0
                - max(
                    -1.0,
                    min(1.0, sum(a * b for a, b in zip(query, vector, strict=True))),
                )
                for _, vector in passages
            ]
        lexical, strong_lexical = self._lexical_scores(question, evidence)
        enriched = [
            replace(item, distance=distances[i], lexical_score=lexical[i])
            for i, item in enumerate(evidence)
        ]
        dense_order = sorted(
            range(len(evidence)), key=lambda i: (distances[i], evidence[i].evidence_id)
        )
        lexical_order = sorted(
            range(len(evidence)), key=lambda i: (-lexical[i], evidence[i].evidence_id)
        )
        if mode == "dense":
            candidates = tuple(enriched[i] for i in dense_order)
            _remaining(deadline)
            response = SearchResult(
                candidates[:5],
                candidates,
                time.perf_counter() - started,
                embedding_seconds,
                revision,
            )
            if include_reviewed_qa:
                response = replace(
                    response,
                    reviewed_qa=self._rank_reviewed_qa(
                        question,
                        passages,
                        query_vector,
                        dimension,
                        audience=reviewed_audience,
                        detail=reviewed_detail,
                        threshold=threshold,
                        deadline=deadline,
                    ),
                )
            return replace(response, elapsed_seconds=time.perf_counter() - started)
        if mode == "lexical":
            order = lexical_order
        else:
            rrf = Counter(
                {i: 1.0 / (60 + rank) for rank, i in enumerate(dense_order, 1)}
            )
            for rank, i in enumerate(lexical_order, 1):
                if lexical[i] > 0:
                    rrf[i] += 1.0 / (60 + rank)
            order = sorted(
                range(len(evidence)),
                key=lambda i: (-rrf[i], distances[i], evidence[i].evidence_id),
            )
        best_distance = min(distances)
        best_lexical = max(lexical, default=0)
        selected: list[Evidence] = []
        seen: set[str] = set()
        used_chars = 0
        for i in order:
            dense_relevant = mode != "lexical" and distances[i] <= min(
                threshold, best_distance + 0.2
            )
            lexical_relevant = strong_lexical[i] and lexical[i] >= best_lexical * 0.35
            if not (dense_relevant or lexical_relevant):
                continue
            item = enriched[i]
            normalized = re.sub(r"\s+", " ", item.text).strip()
            if normalized in seen or used_chars + len(item.text) > max_context_chars:
                continue
            seen.add(normalized)
            selected.append(item)
            used_chars += len(item.text)
            if len(selected) >= max_evidence:
                break
        _remaining(deadline)
        response = SearchResult(
            tuple(selected),
            tuple(enriched[i] for i in order),
            time.perf_counter() - started,
            embedding_seconds,
            revision,
        )
        if include_reviewed_qa:
            response = replace(
                response,
                reviewed_qa=self._rank_reviewed_qa(
                    question,
                    passages,
                    query_vector,
                    dimension,
                    audience=reviewed_audience,
                    detail=reviewed_detail,
                    threshold=threshold,
                    deadline=deadline,
                ),
            )
        return replace(response, elapsed_seconds=time.perf_counter() - started)
