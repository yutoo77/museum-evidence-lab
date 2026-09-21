"""ローカルOllamaのチャットAPIを使う回答生成アダプター。"""

from __future__ import annotations

from typing import Protocol

import httpx

from src.exceptions import GenerationError, OllamaConnectionError, OllamaModelError


class GenerationProvider(Protocol):
    model_name: str

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str: ...


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "think": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code == 404:
                raise OllamaModelError(
                    f"回答生成モデル「{self.model_name}」が見つかりません。",
                    response.text,
                )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("message", {}).get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Ollama returned an empty message")
            return content.strip()
        except OllamaModelError:
            raise
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                "Ollamaに接続できません。Ollamaを起動してから再度お試しください。",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except httpx.TimeoutException as exc:
            raise GenerationError(
                "回答生成が時間内に完了しませんでした。もう一度お試しください。",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise GenerationError(
                "ローカルAIによる処理中にエラーが発生しました。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

    def warmup(self) -> None:
        """デモ前にモデルをメモリへ読み込み、30分間利用可能にする。"""

        self.chat(
            "あなたはローカルAIの起動確認を行います。",
            "READYとだけ返してください。",
            temperature=0.0,
            max_tokens=2,
        )
