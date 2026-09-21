"""Exact-match, explicitly approved FAQ answers tied to current source versions."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import asdict
from datetime import date
from pathlib import Path

from src.lab.contracts import Claim, LabAnswer


def _key(question):
    return " ".join(unicodedata.normalize("NFKC", question).strip().split()).casefold()


class ApprovedFAQ:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS faq (question TEXT PRIMARY KEY, claims TEXT NOT NULL, expires TEXT NOT NULL)"
            )

    def approve(self, question, claims, evidence, *, expires):
        if (
            not _key(question)
            or not claims
            or date.fromisoformat(expires) < date.today()
        ):
            raise ValueError("有効な質問・回答・有効期限が必要です。")
        by_id = {e.evidence_id: e for e in evidence}
        records = []
        for claim in claims:
            if len(claim.references) != 1:
                raise ValueError(
                    "複数の根拠をまとめた説明のFAQ保存には対応していません。"
                )
            source = by_id.get(claim.evidence_id)
            if source is None or not claim.quote or claim.quote not in source.text:
                raise ValueError("根拠の確認できない回答は承認できません。")
            records.append({**asdict(claim), "content_hash": source.content_hash})
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO faq VALUES (?, ?, ?)",
                (_key(question), json.dumps(records, ensure_ascii=False), expires),
            )

    def lookup(self, question, current_passages):
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT claims, expires FROM faq WHERE question = ?", (_key(question),)
            ).fetchone()
        if row is None or date.fromisoformat(row[1]) < date.today():
            return None
        by_id = {p.evidence_id: p for p in current_passages}
        claims, evidence = [], []
        for item in json.loads(row[0]):
            passage = by_id.get(item["evidence_id"])
            if (
                passage is None
                or passage.content_hash != item["content_hash"]
                or item["quote"] not in passage.text
            ):
                return None
            claims.append(Claim(item["text"], item["evidence_id"], item["quote"]))
            if passage not in evidence:
                evidence.append(passage)
        return LabAnswer(
            "answered", "", tuple(claims), tuple(evidence), route="approved_faq"
        )

    def clear(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM faq")
