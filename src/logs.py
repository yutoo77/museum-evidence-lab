"""本文を記録しない開発者向けアプリケーションログ設定。"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_dir: str | Path) -> None:
    root = logging.getLogger()
    if any(getattr(handler, "name", "") == "museum-rag-file" for handler in root.handlers):
        return
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / "app.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.name = "museum-rag-file"
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)
