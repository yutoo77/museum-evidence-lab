"""Shared contracts for retrieval, generation, and reproducible comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.lab.reviewed_qa import ReviewedQA


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source_name: str
    page_number: int | None
    content_hash: str
    distance: float = 1.0
    lexical_score: float = 0.0
    version_id: str = ""


@dataclass(frozen=True)
class SearchResult:
    evidence: tuple[Evidence, ...]
    candidates: tuple[Evidence, ...] = ()
    elapsed_seconds: float = 0.0
    embedding_seconds: float = 0.0
    revision: str = ""
    reviewed_qa: tuple[ReviewedQA, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    content: str
    done_reason: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceCitation:
    """An exact source span, independent of the wording of an explanation."""

    evidence_id: str
    quote: str


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_id: str
    quote: str
    citations: tuple[SourceCitation, ...] = ()

    @property
    def references(self) -> tuple[SourceCitation, ...]:
        """Keep old quotation/FAQ records readable while supporting synthesis."""
        if self.citations:
            return self.citations
        if self.evidence_id and self.quote:
            return (SourceCitation(self.evidence_id, self.quote),)
        return ()


@dataclass(frozen=True)
class ReviewedQAContext:
    """A staff-reviewed Q&A included as context for an explanation draft.

    Inclusion does not prove that the final wording adopted the earlier answer.
    """

    id: int
    question: str
    approved_answer: str
    created_at: str


@dataclass(frozen=True)
class LabAnswer:
    status: str
    message: str
    claims: tuple[Claim, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    route: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    calls: tuple[dict[str, float | int | str], ...] = ()
    issues: tuple[str, ...] = ()
    reviewed_qa_context: tuple[ReviewedQAContext, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(claim.text for claim in self.claims) or self.message
