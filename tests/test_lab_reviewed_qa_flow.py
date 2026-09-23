"""The reviewed knowledge loop stays source-bound and entirely local."""

import json
import socket

from src.lab.contracts import GenerationResult
from src.lab.engine import LabEngine
from src.lab.index import HybridIndex

SOURCE_NAME = "架空の月の展示.txt"
FIRST_SOURCE = (
    "月の満ち欠けは、月と地球と太陽の位置関係が変わり、"
    "地球から見える月の明るい部分が変わるため起こります。"
)
UPDATED_SOURCE = (
    "月の形が変わって見えるのは、太陽に照らされた月の部分のうち"
    "地球から見える範囲が、月と地球と太陽の位置関係で変わるためです。"
)
GENERATED = (
    "月と地球と太陽の位置関係によって、地球から見える月の明るい部分が変わるためです。"
)
FIRST_QUESTION = "月の満ち欠けは、なぜ起こる？"
RELATED_QUESTION = "月の形が毎日違って見えるのはなぜ？"


class FakeEmbeddings:
    model_name = "fixture-embedding"

    def embed_texts(self, texts, **_options):
        return [[1.0, 0.0] if "月" in text else [0.0, 1.0] for text in texts]


class FakeGeneration:
    def __init__(self):
        self.calls = []

    def chat(self, system, user, **options):
        self.calls.append({"system": system, "user": user, "options": options})
        if len(self.calls) == 1:
            content = json.dumps(
                {"answer": [{"sources": ["P1S1"], "text": GENERATED}]},
                ensure_ascii=False,
            )
        else:
            content = '{"verdict":"SUPPORTED"}'
        return GenerationResult(content, "stop", {"model": "fake-local"})


def ask(index, question):
    client = FakeGeneration()
    answer = LabEngine(index, client).answer(question)
    assert answer.status == "answered"
    assert len(client.calls) == 2
    return answer, json.loads(client.calls[0]["user"])


def test_confirm_reuse_stale_and_archive_flow_never_opens_network(
    tmp_path, monkeypatch
):
    def blocked_network(*_args, **_kwargs):
        raise AssertionError("The reviewed Q&A flow attempted a socket connection")

    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)

    index = HybridIndex(
        tmp_path / "index.sqlite3",
        FakeEmbeddings(),
        embedding_digest="sha256:fixture",
    )
    index.register_bytes(SOURCE_NAME, FIRST_SOURCE.encode("utf-8"))
    first_revision = index.revision
    first_answer, first_prompt = ask(index, FIRST_QUESTION)
    assert "reviewed_qa" not in first_prompt
    first_qa = index.save_reviewed_qa(
        FIRST_QUESTION,
        first_answer,
        "太陽に照らされた月の部分のうち、地球から見える範囲が変わるためです。",
        first_revision,
    )

    related_answer, related_prompt = ask(index, RELATED_QUESTION)
    assert (
        related_prompt["reviewed_qa"][0]["approved_answer"] == first_qa.approved_answer
    )
    assert related_answer.reviewed_qa_context[0].id == first_qa.id
    assert (
        related_answer.claims[0].references[0].evidence_id
        == first_answer.evidence[0].evidence_id
    )

    index.register_bytes(SOURCE_NAME, UPDATED_SOURCE.encode("utf-8"))
    assert index.list_reviewed_qa()[0].status == "needs_review"
    updated_answer, updated_prompt = ask(index, RELATED_QUESTION)
    assert "reviewed_qa" not in updated_prompt
    assert updated_answer.reviewed_qa_context == ()

    second_qa = index.save_reviewed_qa(
        RELATED_QUESTION,
        updated_answer,
        updated_answer.text,
        index.revision,
    )
    reused_answer, reused_prompt = ask(index, RELATED_QUESTION)
    assert (
        reused_prompt["reviewed_qa"][0]["approved_answer"] == second_qa.approved_answer
    )
    assert reused_answer.reviewed_qa_context[0].id == second_qa.id

    index.archive_reviewed_qa(second_qa.id)
    archived_answer, archived_prompt = ask(index, RELATED_QUESTION)
    assert "reviewed_qa" not in archived_prompt
    assert archived_answer.reviewed_qa_context == ()
    assert {item.id: item.status for item in index.list_reviewed_qa()} == {
        first_qa.id: "needs_review",
        second_qa.id: "archived",
    }
