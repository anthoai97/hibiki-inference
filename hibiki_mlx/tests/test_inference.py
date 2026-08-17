"""Tests for checking the artifact bundle's weights."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hibiki_mlx.inference import WeightCheckError, read_tensor_manifest, start

MIMI_FILE = "mimi-test.safetensors"
HIBIKI_FILE = "hibiki-test.safetensors"


def write_safetensors(path: Path, header: dict[str, object]) -> None:
    """Write a safetensors file whose header is real and whose payload is empty."""
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def write_bundle(directory: Path) -> None:
    write_safetensors(
        directory / MIMI_FILE,
        {
            "__metadata__": {"format": "pt"},
            "encoder.model.0.conv.conv.weight": {
                "dtype": "F32",
                "shape": [64, 1, 7],
                "data_offsets": [0, 0],
            },
        },
    )
    write_safetensors(
        directory / HIBIKI_FILE,
        {
            "text_emb.weight": {"dtype": "BF16", "shape": [48001, 2048], "data_offsets": [0, 0]},
            "transformer.layers.0.self_attn.out_proj.weight": {
                "dtype": "BF16",
                "shape": [2048, 2048],
                "data_offsets": [0, 0],
            },
        },
    )
    (directory / "config.json").write_text(
        json.dumps({"mimi_name": MIMI_FILE, "moshi_name": HIBIKI_FILE})
    )


class ReadTensorManifestTests(unittest.TestCase):
    def test_reports_each_tensor_dtype_and_shape_without_the_metadata_entry(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / MIMI_FILE
            write_safetensors(
                path,
                {
                    "__metadata__": {"format": "pt"},
                    "quantizer.bias": {"dtype": "F32", "shape": [512], "data_offsets": [0, 0]},
                },
            )

            self.assertEqual(read_tensor_manifest(path), {"quantizer.bias": ("F32", (512,))})

    def test_rejects_a_file_that_is_too_small_to_hold_a_header_length(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / MIMI_FILE
            path.write_bytes(b"\x00\x00")

            with self.assertRaises(WeightCheckError):
                read_tensor_manifest(path)

    def test_rejects_a_truncated_header(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / MIMI_FILE
            path.write_bytes(struct.pack("<Q", 4096) + b'{"a":')

            with self.assertRaises(WeightCheckError):
                read_tensor_manifest(path)


class StartTests(unittest.TestCase):
    def test_checks_the_mimi_weights_before_the_hibiki_weights(self) -> None:
        with TemporaryDirectory() as directory:
            write_bundle(Path(directory))

            mimi, hibiki = start(artifact_directory=directory)

            self.assertEqual(mimi.path.name, MIMI_FILE)
            self.assertEqual(mimi.tensor_count, 1)
            self.assertEqual(mimi.dtypes, ("F32",))
            self.assertEqual(mimi.prefixes, ("encoder",))

            self.assertEqual(hibiki.path.name, HIBIKI_FILE)
            self.assertEqual(hibiki.parameter_count, 48001 * 2048 + 2048 * 2048)
            self.assertEqual(hibiki.dtypes, ("BF16",))
            self.assertEqual(hibiki.prefixes, ("text_emb", "transformer"))

    def test_rejects_a_bundle_whose_configuration_names_a_missing_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            write_bundle(Path(directory))
            (Path(directory) / MIMI_FILE).unlink()

            with self.assertRaises(WeightCheckError):
                start(artifact_directory=directory)

    def test_reports_a_missing_configuration_instead_of_guessing_file_names(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(WeightCheckError):
                start(artifact_directory=directory)
