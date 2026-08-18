"""Tests for publishing a validated quantized artifact bundle."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hibiki_mlx.artifacts import (
    QUANTIZATION_CONFIG_KEY,
    QuantizationSpec,
    publish_quantized_bundle,
)
from hibiki_mlx.upload import main


def write_bundle(directory: Path) -> None:
    (directory / "model.q8.safetensors").write_bytes(b"weights")
    (directory / "mimi.safetensors").write_bytes(b"codec")
    (directory / "tokenizer.model").write_bytes(b"tokenizer")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "moshi_name": "model.q8.safetensors",
                "mimi_name": "mimi.safetensors",
                "tokenizer_name": "tokenizer.model",
                QUANTIZATION_CONFIG_KEY: QuantizationSpec.for_bits(8).as_config(),
            }
        )
    )


class RecordingHubApi:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.uploaded: list[dict[str, object]] = []

    def create_repo(self, **arguments: object) -> None:
        self.created.append(arguments)

    def upload_folder(self, **arguments: object) -> str:
        self.uploaded.append(arguments)
        return "https://huggingface.co/example/hibiki-q8/commit/test"


class UploadTests(unittest.TestCase):
    def test_dry_run_validates_the_bundle_without_contacting_hugging_face(self) -> None:
        with TemporaryDirectory() as temporary:
            artifact_directory = Path(temporary)
            write_bundle(artifact_directory)
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(artifact_directory),
                        "--repo-id",
                        "example/hibiki-q8",
                        "--private",
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("would upload", stdout.getvalue())

    def test_publication_uses_one_hub_adapter_for_the_complete_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            artifact_directory = Path(temporary)
            write_bundle(artifact_directory)
            hub = RecordingHubApi()

            publication = publish_quantized_bundle(
                hub,
                artifact_directory,
                repo_id="example/hibiki-q8",
                private=True,
                revision="main",
                commit_message="Publish Q8",
            )

        self.assertEqual(publication.spec, QuantizationSpec.for_bits(8))
        self.assertEqual(publication.commit, "https://huggingface.co/example/hibiki-q8/commit/test")
        self.assertEqual(
            hub.created,
            [
                {
                    "repo_id": "example/hibiki-q8",
                    "repo_type": "model",
                    "private": True,
                    "exist_ok": True,
                }
            ],
        )
        self.assertEqual(hub.uploaded[0]["folder_path"], str(artifact_directory))
        self.assertEqual(hub.uploaded[0]["commit_message"], "Publish Q8")
