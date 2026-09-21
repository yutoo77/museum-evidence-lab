"""Opt-in operational metadata only; never log questions, answers, or thoughts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


class AuditLog:
    def __init__(self, directory, *, enabled=False, retention_days=7):
        self.directory = Path(directory)
        self.enabled = enabled
        self.retention_days = retention_days

    def save(self, answer):
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        record = {
            "timestamp": now.isoformat(),
            "status": answer.status,
            "route": answer.route,
            "timings": answer.timings,
            "call_count": len(answer.calls),
            "issues": answer.issues,
        }
        with (self.directory / f"{now.date().isoformat()}.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        for path in self.directory.glob("????-??-??.jsonl"):
            try:
                day = datetime.strptime(path.stem, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < (now - timedelta(days=self.retention_days)).date():
                path.unlink()
