"""Measured, fail-closed access to explicitly allowed, local Ollama models."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.lab.contracts import GenerationResult


class LocalModelError(RuntimeError):
    """A local model is unavailable or returned an unusable response."""


class LocalModelSecurityError(LocalModelError):
    """The endpoint or model cannot be verified as an allowed local resource."""


def validate_base_url(base_url: str) -> str:
    """Accept literal loopback addresses only; pin localhost to IPv4 loopback."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise LocalModelSecurityError("Ollamaの接続先が設定されていません。")
    if any(character in base_url for character in ("?", "#", "\\")):
        raise LocalModelSecurityError("OllamaのURLに追加のパスや設定は指定できません。")
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise LocalModelSecurityError("OllamaのURLが正しくありません。") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or host is None
        or any(character.isspace() for character in base_url)
    ):
        raise LocalModelSecurityError("OllamaはPC内の接続先だけを指定してください。")
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise LocalModelSecurityError(
            "Ollamaの接続先はループバックアドレスに限定されています。"
        ) from exc
    if not address.is_loopback:
        raise LocalModelSecurityError("PC外のOllamaには接続できません。")
    authority = f"[{address}]" if address.version == 6 else str(address)
    if port is not None:
        if not 1 <= port <= 65535:
            raise LocalModelSecurityError("Ollamaのポート番号が正しくありません。")
        authority += f":{port}"
    return f"{parsed.scheme}://{authority}"


def _canonical_model(model: str) -> str:
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def validate_allowed_models(models: Sequence[str]) -> tuple[str, ...]:
    if isinstance(models, (str, bytes)) or not models:
        raise LocalModelSecurityError(
            "使用を許可するローカルモデルを指定してください。"
        )
    result = tuple(models)
    for model in result:
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model)
            or "://" in model
            or "cloud" in model.lower()
            or "remote" in model.lower()
            or ".." in model
        ):
            raise LocalModelSecurityError(
                "許可リストにはローカルモデル名だけを指定してください。"
            )
    if len({_canonical_model(model) for model in result}) != len(result):
        raise LocalModelSecurityError("使用を許可するモデル名が重複しています。")
    return result


def _metrics(
    payload: Mapping[str, Any], wall_seconds: float
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {"wall_seconds": wall_seconds}
    for source, target in (
        ("total_duration", "total_seconds"),
        ("load_duration", "load_seconds"),
        ("prompt_eval_duration", "prefill_seconds"),
        ("eval_duration", "decode_seconds"),
    ):
        value = payload.get(source)
        metrics[target] = (
            float(value) / 1_000_000_000
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            else "not_reported"
        )
    for source, target in (
        ("prompt_eval_count", "input_tokens"),
        ("eval_count", "output_tokens"),
        ("prompt_eval_cached_count", "cached_tokens"),
    ):
        value = payload.get(source)
        if source == "prompt_eval_cached_count" and value is None:
            value = payload.get("cache_read_input_tokens")
        metrics[target] = (
            value
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            else "not_reported"
        )
    return metrics


class _LocalOllama:
    def __init__(
        self,
        base_url: str,
        allowed_models: Sequence[str],
        timeout_seconds: float = 90,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 600
        ):
            raise ValueError("通信の待ち時間は1〜600秒で指定してください。")
        self.base_url = validate_base_url(base_url)
        self.allowed_models = validate_allowed_models(allowed_models)
        self.timeout_seconds = timeout_seconds
        self._verified: dict[str, str] = {}
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict:
        try:
            kwargs = {"timeout": timeout_seconds} if timeout_seconds is not None else {}
            response = self._http.request(method, endpoint, json=payload, **kwargs)
            if response.is_redirect:
                raise LocalModelSecurityError(
                    "Ollamaからの別URLへの転送を拒否しました。"
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LocalModelError(
                "PC内のモデル処理が時間内に完了しませんでした。"
            ) from exc
        except httpx.HTTPError as exc:
            # Server error bodies can contain the question or document. Do not surface them.
            raise LocalModelError(
                "PC内のOllamaに接続できないか、処理に失敗しました。"
            ) from exc
        try:
            result = response.json()
        except ValueError as exc:
            raise LocalModelError("Ollamaの応答形式が正しくありません。") from exc
        if not isinstance(result, dict) or result.get("error"):
            raise LocalModelError("Ollamaから有効な応答を受け取れませんでした。")
        return result

    def _inventory(self, timeout_seconds: float | None = None) -> list[dict]:
        models = self._request("GET", "/api/tags", timeout_seconds=timeout_seconds).get(
            "models"
        )
        if not isinstance(models, list) or any(
            not isinstance(model, dict) for model in models
        ):
            raise LocalModelSecurityError("PC内のモデル一覧を確認できませんでした。")
        return models

    def _verify(self, model_name: str, timeout_seconds: float | None = None) -> str:
        deadline = (
            time.perf_counter() + timeout_seconds
            if timeout_seconds is not None
            else None
        )
        if model_name not in self.allowed_models:
            raise LocalModelSecurityError("このモデルは使用を許可されていません。")
        canonical = _canonical_model(model_name)
        matches = [
            model
            for model in self._inventory(timeout_seconds)
            if isinstance(model.get("name"), str)
            and _canonical_model(model["name"]) == canonical
        ]
        if deadline is not None:
            _remaining(deadline)
        if len(matches) != 1:
            raise LocalModelSecurityError(
                "指定されたローカルモデルがPC内に見つかりません。"
            )
        digest = matches[0].get("digest")
        if not isinstance(digest, str) or not re.fullmatch(
            r"(?:sha256:)?[a-fA-F0-9]{64}", digest
        ):
            raise LocalModelSecurityError(
                "ローカルモデルの識別情報を確認できませんでした。"
            )
        if self._verified.get(canonical) == digest:
            return digest
        remaining = _remaining(deadline) if deadline is not None else None
        metadata = self._request(
            "POST", "/api/show", {"model": model_name}, timeout_seconds=remaining
        )
        if deadline is not None:
            _remaining(deadline)
        if any(
            metadata.get(key) for key in ("remote_host", "remote_model", "remote_url")
        ):
            raise LocalModelSecurityError("外部で動作するモデルは使用できません。")
        details = metadata.get("details")
        information = metadata.get("model_info")
        if (
            not isinstance(details, dict)
            or details.get("format") != "gguf"
            or not isinstance(information, dict)
            or not isinstance(information.get("general.architecture"), str)
            or not information["general.architecture"]
        ):
            raise LocalModelSecurityError(
                "モデルがPC内で動くことを確認できませんでした。"
            )
        # A model import can replace the tag while /api/show is in flight.
        # Do not attribute the inspected metadata to a stale digest.
        remaining = _remaining(deadline) if deadline is not None else None
        verified_tags = [
            item
            for item in self._inventory(remaining)
            if isinstance(item.get("name"), str)
            and _canonical_model(item["name"]) == canonical
        ]
        if deadline is not None:
            _remaining(deadline)
        if len(verified_tags) != 1 or verified_tags[0].get("digest") != digest:
            raise LocalModelSecurityError(
                "確認中にモデルが変更されました。モデルの更新完了後にお試しください。"
            )
        self._verified[canonical] = digest
        return digest

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class InstrumentedOllamaClient(_LocalOllama):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        allowed_models: Sequence[str],
        timeout_seconds: float = 90,
        context_length: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if type(context_length) is not int or not 512 <= context_length <= 32768:
            raise ValueError("コンテキスト長は512〜32768で指定してください。")
        if model_name not in allowed_models:
            raise LocalModelSecurityError("この生成モデルは使用を許可されていません。")
        super().__init__(base_url, allowed_models, timeout_seconds, transport)
        self.model_name = model_name
        self.context_length = context_length

    @property
    def model_digest(self) -> str:
        return self._verify(self.model_name)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 500,
        schema: dict | None = None,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
    ) -> GenerationResult:
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("最大出力数は1〜4096で指定してください。")
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("temperatureは0〜2で指定してください。")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise ValueError("モデルへの入力には文字列を指定してください。")
        if schema is not None and not isinstance(schema, dict):
            raise ValueError("応答の構造にはJSON Schemaを指定してください。")
        started = time.perf_counter()
        budget = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if (
            type(budget) not in (int, float)
            or not math.isfinite(budget)
            or not 0 < budget <= 600
        ):
            raise ValueError("通信の待ち時間は0秒を超え600秒以下で指定してください。")
        deadline = started + budget
        digest = self._verify(self.model_name, _remaining(deadline))
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "think": False,
            "keep_alive": "15m",
            "options": {
                "num_ctx": self.context_length,
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if schema is not None:
            payload["format"] = schema
        result, content, first_token_seconds = self._consume_chat(
            payload, deadline, started
        )
        metrics = _metrics(result, time.perf_counter() - started)
        metrics.update(model=self.model_name, digest=digest)
        metrics["first_token_seconds"] = first_token_seconds
        # Construct a fresh result, never retaining the raw response or thinking field.
        return GenerationResult(
            content=content,
            done_reason=str(result.get("done_reason", "unknown")),
            metrics=metrics,
        )

    def _consume_chat(
        self, payload: dict, deadline: float, started: float
    ) -> tuple[dict, str, float | str]:
        parts: list[str] = []
        content_length = 0
        first_token_seconds: float | str = "not_reported"
        final = None
        try:
            with self._http.stream(
                "POST",
                "/api/chat",
                json=payload,
                timeout=_remaining(deadline),
            ) as response:
                if response.is_redirect:
                    raise LocalModelSecurityError(
                        "Ollamaからの別URLへの転送を拒否しました。"
                    )
                response.raise_for_status()
                for line in response.iter_lines():
                    _remaining(deadline)
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise LocalModelError(
                            "モデルの応答形式が正しくありません。"
                        ) from exc
                    if not isinstance(chunk, dict) or chunk.get("error"):
                        raise LocalModelError("モデルの応答を受け取れませんでした。")
                    self._validate_response_model(chunk)
                    message = chunk.get("message")
                    if not isinstance(message, dict) or not isinstance(
                        message.get("content"), str
                    ):
                        raise LocalModelError("モデルの回答形式が正しくありません。")
                    text = message["content"]
                    if text:
                        if first_token_seconds == "not_reported":
                            first_token_seconds = time.perf_counter() - started
                        content_length += len(text)
                        if content_length > 100_000:
                            raise LocalModelError(
                                "モデルの回答が長すぎるため中断しました。"
                            )
                        parts.append(text)
                    # Only the content and final numeric measurements survive this loop.
                    if chunk.get("done") is True:
                        final = {
                            key: value
                            for key, value in chunk.items()
                            if key
                            in {
                                "done_reason",
                                "total_duration",
                                "load_duration",
                                "prompt_eval_duration",
                                "eval_duration",
                                "prompt_eval_count",
                                "eval_count",
                                "cache_read_input_tokens",
                                "prompt_eval_cached_count",
                            }
                        }
                        break
        except httpx.TimeoutException as exc:
            raise LocalModelError(
                "PC内のモデル処理が時間内に完了しませんでした。"
            ) from exc
        except httpx.HTTPError as exc:
            raise LocalModelError("PC内のモデルとの通信に失敗しました。") from exc
        if final is None:
            raise LocalModelError(
                "モデルの回答が途中で切れました。再度お試しください。"
            )
        return final, "".join(parts), first_token_seconds

    def _validate_response_model(self, result: dict) -> None:
        reported = result.get("model")
        if not isinstance(reported, str) or _canonical_model(
            reported
        ) != _canonical_model(self.model_name):
            raise LocalModelSecurityError("応答したモデルが指定と一致しません。")

    def warmup(self) -> dict[str, float | int | str]:
        started = time.perf_counter()
        digest = self.model_digest
        result = self._request(
            "POST",
            "/api/generate",
            {
                "model": self.model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": "15m",
                "options": {"num_ctx": self.context_length},
            },
        )
        self._validate_response_model(result)
        if result.get("done") is not True:
            raise LocalModelError("回答モデルの準備が完了しませんでした。")
        metrics = _metrics(result, time.perf_counter() - started)
        metrics.update(model=self.model_name, digest=digest)
        return metrics


def _remaining(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise LocalModelError("モデル処理の待ち時間を超えました。")
    return remaining


class SafeEmbeddingClient(_LocalOllama):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        allowed_models: Sequence[str],
        timeout_seconds: float = 90,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if model_name not in allowed_models:
            raise LocalModelSecurityError("この検索モデルは使用を許可されていません。")
        super().__init__(base_url, allowed_models, timeout_seconds, transport)
        self.model_name = model_name
        self.last_metrics: dict[str, float | int | str] = {}

    @property
    def model_digest(self) -> str:
        return self._verify(self.model_name)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        if isinstance(texts, (str, bytes)) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("検索用の入力には文字列の一覧を指定してください。")
        self.last_metrics = {}
        started = time.perf_counter()
        budget = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if (
            type(budget) not in (int, float)
            or not math.isfinite(budget)
            or not 0 < budget <= 600
        ):
            raise ValueError("通信の待ち時間は0秒を超え600秒以下で指定してください。")
        deadline = started + budget
        if not texts:
            return []
        digest = self._verify(self.model_name, _remaining(deadline))
        result = self._request(
            "POST",
            "/api/embed",
            {
                "model": self.model_name,
                "input": list(texts),
                "truncate": False,
                "keep_alive": "15m",
            },
            timeout_seconds=_remaining(deadline),
        )
        _remaining(deadline)
        reported = result.get("model")
        if not isinstance(reported, str) or _canonical_model(
            reported
        ) != _canonical_model(self.model_name):
            raise LocalModelSecurityError("応答した検索モデルが指定と一致しません。")
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise LocalModelError("検索モデルから必要な結果を受け取れませんでした。")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise LocalModelError("検索モデルの出力が正しくありません。")
            if dimension is not None and len(vector) != dimension:
                raise LocalModelError("検索モデルの出力の大きさが一致しません。")
            dimension = len(vector)
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in vector
            ):
                raise LocalModelError("検索モデルの出力に無効な数値があります。")
        _remaining(deadline)
        self.last_metrics = _metrics(result, time.perf_counter() - started)
        self.last_metrics.update(model=self.model_name, digest=digest)
        return [[float(value) for value in vector] for vector in vectors]

    def warmup(self) -> dict[str, float | int | str]:
        self.embed_texts(["検索の準備"])
        return dict(self.last_metrics)


def inspect_health(
    base_url: str,
    allowed_models: Sequence[str],
    timeout_seconds: float = 5,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Inspect only local metadata; no user questions or documents are sent."""
    with _LocalOllama(base_url, allowed_models, timeout_seconds, transport) as client:
        version = client._request("GET", "/api/version").get("version")
        inventory = client._inventory()
        installed = {
            _canonical_model(item["name"]): item
            for item in inventory
            if isinstance(item.get("name"), str)
        }
        models = []
        for name in allowed_models:
            item = installed.get(_canonical_model(name))
            status: dict[str, Any] = {
                "name": name,
                "installed": item is not None,
                "local": False,
            }
            if item is not None:
                try:
                    status["digest"] = client._verify(name)
                    status["local"] = True
                except LocalModelError:
                    status["error"] = "ローカルモデルとして確認できませんでした。"
            models.append(status)
        return {
            "status": "connected",
            "base_url": client.base_url,
            "version": str(version or "unknown"),
            "models": models,
        }
