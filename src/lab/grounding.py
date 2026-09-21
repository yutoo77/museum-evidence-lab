"""Shared, deliberately narrow provenance guards; not semantic verification."""

from __future__ import annotations

import re

from src.lab.contracts import Evidence

INJECTION = re.compile(
    r"ignore.{0,30}(instruction|previous)|system\s*prompt|"
    r"指示.{0,12}無視|命令.{0,12}従|API.?キー|外部.{0,12}送信|秘密.{0,12}出力",
    re.IGNORECASE,
)
REMOTE = re.compile(r"(?:https?|ftp)://|www\.|!\[|<\s*(?:img|iframe|script)", re.I)
NUMBER = re.compile(
    r"\d+(?:[.,:]\d+)*(?:時|分|円|日|年|月|秒|度|℃|%|％|cm|mm|km|kg|m|g)?"
)
QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_-])(\d+(?:\.\d+)?)\s*(人|席|円|分|秒|℃|cm|mm|km|kg)(?![A-Za-z])"
)


def source_sentences(evidence: tuple[Evidence, ...]) -> dict[str, tuple[Evidence, str]]:
    """Stable exact spans; the model never supplies quotation text itself."""
    result = {}
    for passage_index, passage in enumerate(evidence, 1):
        for sentence_index, match in enumerate(
            re.finditer(r"[^。！？\n]+[。！？]?", passage.text), 1
        ):
            sentence = match.group().strip()
            if len(sentence) < 3 or len(sentence) > 500 or INJECTION.search(sentence):
                continue
            result[f"P{passage_index}S{sentence_index}"] = (passage, sentence)
    return result


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def quantity_signature(text):
    """Only detects identical wording with changed explicit numeric quantities."""
    compact = re.sub(r"\s+", "", text)
    if not QUANTITY.search(compact):
        return None
    signature = QUANTITY.sub(lambda match: "#" + match[2], compact)
    return signature if len(signature) >= 12 else None
