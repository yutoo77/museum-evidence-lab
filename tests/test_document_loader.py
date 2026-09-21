from __future__ import annotations

import fitz
import pytest

from src.document_loader import DocumentLoader
from src.exceptions import EmptyDocumentError


@pytest.mark.parametrize(
    ("filename", "content", "expected_type"),
    [
        ("案内.txt", "展示室は午前9時に開きます。", "txt"),
        ("説明.md", "# 展示\n\n光について説明します。", "md"),
    ],
)
def test_load_txt_and_markdown(filename, content, expected_type):
    loaded = DocumentLoader().load_bytes(filename, content.encode("utf-8"))

    assert loaded.file_type == expected_type
    assert loaded.pages[0].page_number is None
    assert content.lstrip("\ufeff") in loaded.pages[0].text


def test_pdf_keeps_one_based_page_numbers():
    pdf = fitz.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "First page")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page")
    data = pdf.tobytes()
    pdf.close()

    loaded = DocumentLoader().load_bytes("guide.pdf", data)

    assert loaded.page_count == 2
    assert [page.page_number for page in loaded.pages] == [1, 2]
    assert "First page" in loaded.pages[0].text
    assert "Second page" in loaded.pages[1].text


def test_image_only_pdf_is_reported_as_empty():
    pdf = fitz.open()
    pdf.new_page()
    data = pdf.tobytes()
    pdf.close()

    with pytest.raises(EmptyDocumentError) as error:
        DocumentLoader().load_bytes("scan.pdf", data)

    assert "画像のみ" in error.value.user_message
