from dataclasses import replace

from src.lab.contracts import Claim, Evidence, LabAnswer
from src.lab.engine import LabEngine, decode_selection, validate_selection

SOURCE = Evidence("id", "銀河ラボの見学料金は300円です。", "guide.txt", None, "a")


def test_duplicate_wire_keys_cannot_erase_a_refusal():
    answer = decode_selection(
        '{"selection":["INSUFFICIENT"],"selection":["P1S1"]}', (SOURCE,), concise=False
    )
    assert answer.status == "needs_review"
    assert answer.issues == ("duplicate_json_key",)


def test_duplicate_nested_keys_cannot_replace_a_citation():
    answer = decode_selection(
        '{"selection":[{"sentence_id":"CONFLICT","sentence_id":"P1S1","text":"説明"}]}',
        (SOURCE,),
        concise=True,
    )
    assert answer.status == "needs_review"


def test_whitespace_difference_is_not_a_quantity_conflict():
    second = replace(SOURCE, content_hash="b", text="銀河ラボの見学料金は300 円です。")
    answer = validate_selection(
        '{"status":"answered","claims":[{"sentence_id":"P1S1"}]}',
        (SOURCE, second),
        concise=False,
    )
    assert answer.status == "answered"


def test_faq_source_revision_is_checked_after_lookup():
    class Index:
        revision = "before"

        def export_passages(self):
            return (SOURCE,)

    index = Index()

    class FAQ:
        def lookup(self, question, evidence):
            index.revision = "after"
            return LabAnswer(
                "answered",
                "",
                (Claim(SOURCE.text, "id", SOURCE.text),),
                evidence,
                route="approved_faq",
            )

    answer = LabEngine(index, object(), faq=FAQ()).answer("見学料金は？", mode="quoted")
    assert answer.status == "needs_review"
    assert answer.issues == ("source_changed",)
    assert not answer.claims
