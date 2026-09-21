from pathlib import Path

import pytest

from tools.check_publication import inspect_file


@pytest.mark.parametrize(
    "name,rule",
    [
        ("data/lab/index.sqlite3", "runtime_data"),
        ("outputs/report.json", "runtime_or_private_directory"),
        (".env.local", "environment_secrets"),
        (".streamlit/secrets.toml", "streamlit_secrets"),
        ("model.gguf", "private_or_model_file"),
        ("../outside.txt", "unsafe_path"),
        ("missing.txt", "missing_file"),
    ],
)
def test_private_publication_paths_are_rejected(tmp_path, name, rule):
    assert rule in inspect_file(tmp_path, name)


@pytest.mark.parametrize(
    "body,rule",
    [
        ("ghp_" + "a" * 36, "github_token"),
        ("AKIA" + "A" * 16, "aws_access_key"),
        ("sk-proj-" + "a" * 48, "openai_key"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private_key"),
        ("C:" + "/Users/" + "someone/private.txt", "personal_home_path"),
    ],
)
def test_secret_signatures_are_reported_without_values(tmp_path, body, rule):
    (tmp_path / "example.txt").write_text(body, encoding="utf-8")
    result = inspect_file(tmp_path, "example.txt")
    assert result == [rule]
    assert body not in repr(result)


def test_public_sources_and_placeholder_directories_are_allowed(tmp_path):
    for name in ("README.md", "data/documents/.gitkeep", ".env.example"):
        path = tmp_path / Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public example", encoding="utf-8")
        assert inspect_file(tmp_path, name) == []
