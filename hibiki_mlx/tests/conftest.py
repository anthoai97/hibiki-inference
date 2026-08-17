"""Synthetic artifact bundles for quick tests.

Quick tests must run without network access and without the ~4 GB of released
weights, so these fixtures rebuild a bundle from `tests/data/weight_manifest_*`
-- a manifest captured off the released artifacts by `tools/capture_weight_manifest.py`
that records tensor names, dtypes and shapes but no weight values.

That capture is deliberately an *independent* oracle. `hibiki_mlx` derives the
parameters it expects from `config.json`; these fixtures replay what the real
artifacts actually contain. A test that loads an unmodified bundle therefore
proves the derivation agrees with the release, rather than agreeing with itself.

The safetensors files written here carry a real header and no tensor data. The
loader inspects headers only -- it never reads weights, so it can validate a
3.6 GB artifact in milliseconds -- which is what makes kilobyte fixtures viable.
"""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from hibiki_mlx.trust import PINNED_REVISION

DATA_DIR = Path(__file__).parent / "data"

# Byte width of each safetensors dtype this project encounters.
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "U8": 1}

Manifest = dict[str, tuple[str, list[int]]]


def write_safetensors_header(path: Path, manifest: Manifest) -> None:
    """Write a safetensors file containing a valid header and no tensor data."""
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in manifest.items():
        size = _DTYPE_BYTES[dtype]
        for extent in shape:
            size *= extent
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)


class BundleBuilder:
    """A mutable artifact bundle that tests perturb before writing to disk."""

    def __init__(self, root: Path, captured: dict[str, Any]) -> None:
        self.root = root
        self.config: dict[str, Any] = deepcopy(captured["config"])
        self.lm: Manifest = {
            name: (dtype, list(shape)) for name, (dtype, shape) in captured["weights"]["lm"].items()
        }
        self.mimi: Manifest = {
            name: (dtype, list(shape))
            for name, (dtype, shape) in captured["weights"]["mimi"].items()
        }
        self.tokenizer_bytes = b"synthetic sentencepiece model"
        self.extra_files: dict[str, bytes] = {}

    def write(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(self.config, indent=1))
        write_safetensors_header(self.root / self.config["moshi_name"], self.lm)
        write_safetensors_header(self.root / self.config["mimi_name"], self.mimi)
        (self.root / self.config["tokenizer_name"]).write_bytes(self.tokenizer_bytes)
        for name, payload in self.extra_files.items():
            (self.root / name).write_bytes(payload)
        return self.root


@pytest.fixture(scope="session")
def captured_manifest() -> dict[str, Any]:
    path = DATA_DIR / f"weight_manifest_{PINNED_REVISION[:7]}.json"
    captured: dict[str, Any] = json.loads(path.read_text())
    return captured


@pytest.fixture
def builder(tmp_path: Path, captured_manifest: dict[str, Any]) -> BundleBuilder:
    return BundleBuilder(tmp_path / "bundle", captured_manifest)
