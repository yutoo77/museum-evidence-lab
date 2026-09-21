from __future__ import annotations

from src.models import SearchHit
from src.retrieval import evaluate_relevance


def hit(distance: float) -> SearchHit:
    return SearchHit("本文", distance, "資料.txt", None, 1, "doc", "version")


def test_relevance_uses_the_minimum_distance():
    minimum, relevant = evaluate_relevance([hit(0.72), hit(0.31), hit(0.55)], 0.45)

    assert minimum == 0.31
    assert relevant is True


def test_no_hits_are_outside_the_registered_materials():
    minimum, relevant = evaluate_relevance([], 0.45)

    assert minimum is None
    assert relevant is False
