"""Check the Git publication surface; never print suspected secret contents.

This narrow scanner is a release guard, not a comprehensive secret/PII audit.
Ignored local data is not scanned or transmitted. CI scans tracked files only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

FORBIDDEN_DIRS = {".venv", ".git", "outputs", "work", "__pycache__", "node_modules"}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".gguf",
    ".safetensors",
    ".pem",
    ".key",
}
PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
    ),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openai_key": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
    "personal_home_path": re.compile(rb"[A-Za-z]:[/\\]Users[/\\][A-Za-z0-9_.-]+[/\\]"),
}


def inspect_file(root: Path, name: str) -> list[str]:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        return ["unsafe_path"]
    if any(part.lower() in FORBIDDEN_DIRS for part in relative.parts):
        return ["runtime_or_private_directory"]
    if relative.parts[0].lower() == "data" and relative.name != ".gitkeep":
        return ["runtime_data"]
    if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
        return ["private_or_model_file"]
    if relative.name.lower().startswith(".env") and relative.name != ".env.example":
        return ["environment_secrets"]
    if relative.as_posix().lower() == ".streamlit/secrets.toml":
        return ["streamlit_secrets"]
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        return ["linked_or_external_file"]
    if not path.is_file():
        return ["missing_file"]
    if path.stat().st_size > 2 * 1024 * 1024:
        return ["oversize_review_required"]
    body = path.read_bytes()
    return [label for label, pattern in PATTERNS.items() if pattern.search(body)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    command = ["git", "ls-files", "--cached", "-z"]
    if args.include_untracked:
        command.extend(["--others", "--exclude-standard"])
    result = subprocess.run(command, cwd=root, check=True, capture_output=True)
    names = sorted(set(result.stdout.decode("utf-8").rstrip("\0").split("\0")))
    if not names or names == [""]:
        raise SystemExit("No publication files found")
    findings = [
        {"path": name, "rules": rules}
        for name in names
        if (rules := inspect_file(root, name))
    ]
    print(
        json.dumps(
            {"files_checked": len(names), "findings": findings}, ensure_ascii=True
        )
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
