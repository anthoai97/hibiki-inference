"""Reading a safetensors header without reading its weights.

Validation needs names, dtypes and shapes only. Reading just the header keeps a
3.6 GB artifact verifiable in milliseconds, and keeps the loader from allocating
anything before the bundle has been accepted.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .errors import ArtifactContentError

# A header longer than this is not a header we wrote or a release we know.
_MAX_HEADER_BYTES = 64 * 1024 * 1024

TensorManifest = dict[str, tuple[str, tuple[int, ...]]]


def read_tensor_manifest(path: Path) -> TensorManifest:
    """Return every tensor's dtype and shape, keyed by parameter name."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise ArtifactContentError(
                    f"{path} is too small to be a safetensors file: it has no header length."
                )
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len == 0 or header_len > _MAX_HEADER_BYTES:
                raise ArtifactContentError(
                    f"{path} declares an implausible safetensors header of "
                    f"{header_len} bytes, so it is not a readable artifact."
                )
            encoded = handle.read(header_len)
    except OSError as error:
        raise ArtifactContentError(f"{path} could not be read: {error}.") from error

    if len(encoded) < header_len:
        raise ArtifactContentError(
            f"{path} is truncated: its header claims {header_len} bytes but only "
            f"{len(encoded)} are present."
        )
    try:
        header = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ArtifactContentError(
            f"{path} has a safetensors header that is not valid JSON: {error}."
        ) from error
    if not isinstance(header, dict):
        raise ArtifactContentError(f"{path} has a safetensors header that is not an object.")

    header.pop("__metadata__", None)
    manifest: TensorManifest = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or "dtype" not in entry or "shape" not in entry:
            raise ArtifactContentError(
                f"{path} describes tensor {name!r} without a dtype and shape."
            )
        manifest[name] = (str(entry["dtype"]), tuple(int(extent) for extent in entry["shape"]))
    return manifest
