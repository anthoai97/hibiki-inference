"""Tests for downloading the pinned model artifacts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from hibiki_mlx.download import MODEL_FILES, MODEL_REPOSITORY, MODEL_REVISION, download_model


class DownloadModelTests(unittest.TestCase):
    def test_fetches_each_required_file_into_destination(self) -> None:
        with TemporaryDirectory() as directory:
            artifact_directory = Path(directory)
            with patch("hibiki_mlx.download._hf_hub_download") as hub_download:
                destination = download_model(destination=artifact_directory)

            self.assertEqual(destination, artifact_directory)
            self.assertTrue(artifact_directory.is_dir())
            self.assertEqual(
                hub_download.call_args_list,
                [
                    call(
                        repo_id=MODEL_REPOSITORY,
                        filename=filename,
                        revision=MODEL_REVISION,
                        local_dir=artifact_directory,
                    )
                    for filename in MODEL_FILES
                ],
            )

    def test_allows_an_explicit_revision(self) -> None:
        with TemporaryDirectory() as directory:
            with patch("hibiki_mlx.download._hf_hub_download") as hub_download:
                download_model(destination=Path(directory), revision="test-revision")

            self.assertTrue(
                all(call_args.kwargs["revision"] == "test-revision" for call_args in hub_download.call_args_list)
            )
