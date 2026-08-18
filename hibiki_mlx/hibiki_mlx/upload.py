"""Publish a validated quantized Hibiki artifact bundle to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import publish_quantized_bundle, validate_quantized_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, e.g. user/hibiki-q8")
    visibility = parser.add_mutually_exclusive_group(required=True)
    visibility.add_argument("--private", action="store_true", help="create a private repository")
    visibility.add_argument("--public", action="store_true", help="create a public repository")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload quantized Hibiki artifact bundle")
    parser.add_argument("--dry-run", action="store_true", help="validate without contacting Hugging Face")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    artifact_directory = arguments.artifact_directory.resolve()
    if arguments.dry_run:
        spec = validate_quantized_bundle(artifact_directory)
        print(
            f"would upload Q{spec.bits} bundle from {artifact_directory} "
            f"to {arguments.repo_id}@{arguments.revision}"
        )
        return 0

    from huggingface_hub import HfApi

    publication = publish_quantized_bundle(
        HfApi(),
        artifact_directory,
        repo_id=arguments.repo_id,
        private=arguments.private,
        revision=arguments.revision,
        commit_message=arguments.commit_message,
    )
    print(f"uploaded Q{publication.spec.bits} bundle to {publication.commit}")
    return 0
