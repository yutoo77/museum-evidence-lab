"""文章を検索に適した長さへ、ページをまたがず分割する。"""

from __future__ import annotations

import re

from src.exceptions import EmptyDocumentError
from src.models import LoadedDocument, TextChunk

_BOUNDARIES = "。！？!?\n"


def _normalized(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _choose_end(text: str, start: int, target: int, chunk_size: int) -> int:
    if target >= len(text):
        return len(text)
    minimum = start + max(1, chunk_size // 2)
    for position in range(target, minimum - 1, -1):
        if text[position - 1] in _BOUNDARIES:
            return position
    return target


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """文末・改行を優先しつつ、指定文字数でスライド分割する。"""

    if chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")
    source = _normalized(text)
    if not source:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(source):
        target = min(start + chunk_size, len(source))
        end = _choose_end(source, start, target, chunk_size)
        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(source):
            break
        next_start = max(0, end - overlap)
        while next_start < end and source[next_start].isspace():
            next_start += 1
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int,
    overlap: int,
) -> tuple[TextChunk, ...]:
    chunks: list[TextChunk] = []
    chunk_number = 1
    for page in document.pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(
                TextChunk(
                    text=text,
                    page_number=page.page_number,
                    chunk_number=chunk_number,
                )
            )
            chunk_number += 1
    if not chunks:
        raise EmptyDocumentError(
            "資料内に検索用として登録できる文章がありませんでした。",
            f"No chunks generated from {document.source_name}",
        )
    return tuple(chunks)
