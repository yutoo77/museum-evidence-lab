"""質問応答と職員フィードバックを後方互換なJSONLで保存する。"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models import AnswerResult

_WRITE_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def answer_result_to_event(result: AnswerResult) -> dict[str, Any]:
    return {
        "event_type": "interaction",
        "interaction_id": result.interaction_id,
        "timestamp": result.timestamp,
        "question": result.question,
        "answer": result.answer,
        "status": result.status.value,
        "sources": [
            {
                "source_name": hit.source_name,
                "page_number": hit.page_number,
                "chunk_number": hit.chunk_number,
                "distance": hit.distance,
            }
            for hit in result.evidence
        ],
        "min_distance": result.min_distance,
        "distance_threshold": result.distance_threshold,
        "answerability": result.answerability,
        "visitor_profile": result.visitor_profile,
        "explanation_scene": result.explanation_scene,
        "response_language": result.response_language,
        "embedding_model": result.embedding_model,
        "generation_model": result.generation_model,
        "elapsed_seconds": result.elapsed_seconds,
        "error_detail": result.error_detail,
        "feedback": None,
    }


class InteractionLogStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: dict[str, Any]) -> None:
        serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with _WRITE_LOCK:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")

    def save_interaction(self, result: AnswerResult) -> None:
        self._append(answer_result_to_event(result))

    def save_feedback(
        self,
        result: AnswerResult,
        rating: str,
        comment: str,
    ) -> None:
        event = answer_result_to_event(result)
        event.update(
            {
                "event_type": "feedback",
                "timestamp": utc_now(),
                "feedback": {
                    "rating": rating,
                    "comment": comment.strip(),
                    "saved_at": utc_now(),
                },
            }
        )
        self._append(event)

    def read_events(self) -> tuple[list[dict[str, Any]], int]:
        if not self.path.exists():
            return [], 0
        events: list[dict[str, Any]] = []
        corrupted = 0
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return [], 1
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL event is not an object")
                events.append(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                corrupted += 1
        return events, corrupted

    def list_interactions(self) -> tuple[list[dict[str, Any]], int]:
        events, corrupted = self.read_events()
        interactions: dict[str, dict[str, Any]] = {}
        for event in events:
            interaction_id = str(event.get("interaction_id", "")).strip()
            if not interaction_id:
                continue
            event_type = event.get("event_type", "interaction")
            if event_type == "feedback":
                base = interactions.setdefault(interaction_id, dict(event))
                base["feedback"] = event.get("feedback")
                base["feedback_timestamp"] = event.get("timestamp")
            else:
                prior_feedback = interactions.get(interaction_id, {}).get("feedback")
                interactions[interaction_id] = dict(event)
                if prior_feedback:
                    interactions[interaction_id]["feedback"] = prior_feedback
        ordered = sorted(
            interactions.values(),
            key=lambda item: str(item.get("timestamp", "")),
            reverse=True,
        )
        return ordered, corrupted
