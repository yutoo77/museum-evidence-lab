"""ローカルOllamaの埋め込みAPIを呼び出す。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from src.exceptions import EmbeddingError, OllamaConnectionError, OllamaModelError


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    """`POST /api/embed`を使う最小限のOllamaクライアント。"""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120,
        batch_size: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

    def list_models(self) -> list[str]:
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.timeout_seconds, 10),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaConnectionError(
                "Ollamaに接続できません。Ollamaが起動しているか確認してください。",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return [
            str(model.get("name") or model.get("model"))
            for model in payload.get("models", [])
            if model.get("name") or model.get("model")
        ]

    def model_is_available(self) -> bool:
        expected = self.model_name.split(":", 1)[0]
        return any(name.split(":", 1)[0] == expected for name in self.list_models())

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                response = httpx.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model_name,
                        "input": batch,
                        "truncate": False,
                    },
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 404:
                    raise OllamaModelError(
                        f"検索用モデル「{self.model_name}」が見つかりません。",
                        response.text,
                    )
                response.raise_for_status()
                payload = response.json()
                batch_embeddings = payload.get("embeddings")
                if not isinstance(batch_embeddings, list) or len(batch_embeddings) != len(
                    batch
                ):
                    raise ValueError("Ollama returned an unexpected embedding count")
                embeddings.extend(
                    [[float(value) for value in vector] for vector in batch_embeddings]
                )
            except OllamaModelError:
                raise
            except httpx.ConnectError as exc:
                raise OllamaConnectionError(
                    "Ollamaに接続できません。Ollamaを起動してから、もう一度お試しください。",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            except httpx.TimeoutException as exc:
                raise EmbeddingError(
                    "資料の検索準備が時間内に完了しませんでした。もう一度お試しください。",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                raise EmbeddingError(
                    "資料の検索準備中にエラーが発生しました。",
                    f"{type(exc).__name__}: {exc}",
                ) from exc

        if not embeddings or any(not vector for vector in embeddings):
            raise EmbeddingError(
                "資料の検索準備に失敗しました。",
                "Ollama returned an empty embedding vector.",
            )
        dimension = len(embeddings[0])
        if any(len(vector) != dimension for vector in embeddings):
            raise EmbeddingError(
                "資料の検索準備に失敗しました。",
                "Embedding dimensions are inconsistent.",
            )
        return embeddings
