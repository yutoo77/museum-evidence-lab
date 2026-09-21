"""Local preparation and metadata inspection, with no automatic model downloads."""

import argparse
import json
from pathlib import Path

from src.lab.client import inspect_health
from src.lab.factory import build_lab
from src.lab.settings import load_lab_settings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "prepare", "sample"])
    parser.add_argument("--config", default="lab_config.yaml")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    settings = load_lab_settings(args.config)
    if args.action == "status":
        print(
            json.dumps(
                inspect_health(
                    settings.ollama.base_url, settings.ollama.allowed_models
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    lab = build_lab(args.config, generation_model=args.model)
    try:
        if args.action == "sample":
            for path in sorted(
                (Path(args.config).resolve().parent / "sample_docs").glob("*.txt")
            ):
                record = lab.index.register_bytes(
                    path.name, path.read_bytes(), approved=True
                )
                print(
                    json.dumps(
                        {"registered": record["source_name"]}, ensure_ascii=False
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {"embedding_preparation": lab.embedding.warmup()}, ensure_ascii=False
            ),
            flush=True,
        )
        print(
            json.dumps(
                {"generation_preparation": lab.client.warmup()}, ensure_ascii=False
            ),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "ready": True,
                    "model": lab.client.model_name,
                    "documents": len(lab.index.list_documents()),
                }
            ),
            flush=True,
        )
        return 0
    finally:
        lab.close()


if __name__ == "__main__":
    raise SystemExit(main())
