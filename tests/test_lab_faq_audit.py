import json
from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.lab.audit import AuditLog
from src.lab.contracts import Claim, Evidence, LabAnswer
from src.lab.faq import ApprovedFAQ


def test_faq_requires_exact_question_and_active_source_version(tmp_path):
    evidence = Evidence("id", "開館は10時です。", "guide.txt", None, "hash")
    claim = Claim("開館は10時です。", "id", "開館は10時です。")
    faq = ApprovedFAQ(tmp_path / "faq.db")
    faq.approve(
        "何時に開館？",
        (claim,),
        (evidence,),
        expires=(date.today() + timedelta(days=1)).isoformat(),
    )
    assert faq.lookup("何時に開館？", (evidence,)).route == "approved_faq"
    assert faq.lookup("何時に閉館？", (evidence,)) is None
    assert faq.lookup("何時に開館？", (replace(evidence, content_hash="new"),)) is None
    assert faq.lookup("何時に開館？", ()) is None
    faq.clear()
    assert faq.lookup("何時に開館？", (evidence,)) is None


def test_faq_rejects_unverified_quote_or_expired_approval(tmp_path):
    faq = ApprovedFAQ(tmp_path / "faq.db")
    with pytest.raises(ValueError):
        faq.approve("質問", (), (), expires="2000-01-01")
    with pytest.raises(ValueError):
        faq.approve(
            "質問", (Claim("回答", "missing", "原文"),), (), expires="2099-01-01"
        )


def test_logging_is_off_by_default_and_never_writes_content(tmp_path):
    answer = LabAnswer(
        "answered",
        "PRIVATE_QUESTION",
        (Claim("PRIVATE_ANSWER", "id", "PRIVATE_SOURCE"),),
        route="quoted",
    )
    AuditLog(tmp_path / "off").save(answer)
    assert not (tmp_path / "off").exists()
    AuditLog(tmp_path / "on", enabled=True).save(answer)
    text = next((tmp_path / "on").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "PRIVATE" not in text
    assert json.loads(text)["status"] == "answered"
