"""デモ用架空資料の一括登録。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.document_service import DocumentRegistrationService
from src.exceptions import RAGApplicationError
from src.models import RegistrationResult

DEMO_DOCUMENT_FILENAMES = (
    "moon_phase.txt",
    "solar_system.txt",
    "black_hole.txt",
    "planetarium_guide.txt",
    "museum_faq.txt",
)


@dataclass(frozen=True, slots=True)
class DemoRegistrationSummary:
    results: tuple[RegistrationResult, ...]
    errors: tuple[str, ...]


def register_demo_documents(
    service: DocumentRegistrationService,
    sample_directory: str | Path,
) -> DemoRegistrationSummary:
    directory = Path(sample_directory)
    results: list[RegistrationResult] = []
    errors: list[str] = []
    for filename in DEMO_DOCUMENT_FILENAMES:
        path = directory / filename
        try:
            results.append(service.register_bytes(filename, path.read_bytes()))
        except Exception as exc:
            message = exc.user_message if isinstance(exc, RAGApplicationError) else str(exc)
            errors.append(f"{filename}: {message}")
    return DemoRegistrationSummary(tuple(results), tuple(errors))
