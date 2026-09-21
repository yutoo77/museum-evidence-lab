from __future__ import annotations

from typing import Any

from src.generation import OllamaChatClient


class _SuccessfulResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"message": {"content": "READY"}}


def test_warmup_keeps_model_loaded_for_demo(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: float):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return _SuccessfulResponse()

    monkeypatch.setattr("src.generation.httpx.post", fake_post)
    client = OllamaChatClient("http://127.0.0.1:11434", "qwen3:1.7b", 120)

    client.warmup()

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["json"]["keep_alive"] == "30m"
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["num_predict"] == 2
