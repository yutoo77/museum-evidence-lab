"""PDF、txt、mdを安全に読み込み、ページ単位データへ変換する。"""

from __future__ import annotations

from pathlib import Path

from src.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from src.models import LoadedDocument, PageText

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentLoader:
    """登録資料をアプリ共通形式へ読み込む。"""

    def load_path(self, path: str | Path) -> LoadedDocument:
        source = Path(path)
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise DocumentReadError(
                "資料ファイルを読み込めませんでした。",
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return self.load_bytes(source.name, data)

    def load_bytes(self, filename: str, data: bytes) -> LoadedDocument:
        source_name = Path(filename).name
        extension = Path(source_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedDocumentError(
                "このファイル形式には対応していません。PDF、txt、mdをご利用ください。",
                f"Unsupported extension: {extension or '<none>'}",
            )
        if not data:
            raise EmptyDocumentError(
                "資料が空のため登録できませんでした。",
                f"Empty file: {source_name}",
            )
        if extension == ".pdf":
            return self._load_pdf(source_name, data)
        return self._load_text(source_name, extension.lstrip("."), data)

    @staticmethod
    def _decode_text(data: bytes) -> str:
        failures: list[str] = []
        for encoding in ("utf-8-sig", "cp932"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError as exc:
                failures.append(f"{encoding}: {exc}")
        raise DocumentReadError(
            "文字コードを判定できず、資料を読み込めませんでした。UTF-8で保存し直してください。",
            " | ".join(failures),
        )

    def _load_text(self, source_name: str, file_type: str, data: bytes) -> LoadedDocument:
        text = self._decode_text(data).replace("\x00", "").strip()
        if not text:
            raise EmptyDocumentError(
                "資料内に登録できる文章がありませんでした。",
                f"No usable text in {source_name}",
            )
        return LoadedDocument(
            source_name=source_name,
            file_type=file_type,
            pages=(PageText(text=text, page_number=None),),
            page_count=1,
        )

    @staticmethod
    def _load_pdf(source_name: str, data: bytes) -> LoadedDocument:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - 依存関係不備時のみ
            raise DocumentReadError(
                "PDF読込機能を利用できません。管理者に連絡してください。",
                "PyMuPDF is not installed.",
            ) from exc

        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                page_count = pdf.page_count
                pages = tuple(
                    PageText(
                        text=page.get_text("text", sort=True).replace("\x00", "").strip(),
                        page_number=index + 1,
                    )
                    for index, page in enumerate(pdf)
                )
        except Exception as exc:
            raise DocumentReadError(
                "PDFを読み込めませんでした。ファイルが破損していないか確認してください。",
                f"{type(exc).__name__}: {exc}",
            ) from exc

        if page_count == 0 or not any(page.text for page in pages):
            raise EmptyDocumentError(
                "PDFから文章を取り出せませんでした。画像のみのPDFには現在対応していません。",
                f"PDF contains no extractable text: {source_name}, pages={page_count}",
            )

        empty_pages = sum(not page.text for page in pages)
        warning = ""
        if empty_pages:
            warning = f"{empty_pages}ページは文章を取り出せなかったため、登録対象外です。"
        return LoadedDocument(
            source_name=source_name,
            file_type="pdf",
            pages=pages,
            page_count=page_count,
            warning=warning,
        )
