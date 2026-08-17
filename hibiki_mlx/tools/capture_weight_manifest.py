"""Capture the parameter manifest of a real artifact bundle as test data.

This tool lives outside the production package. It records only tensor names,
dtypes and shapes -- never weight values -- so the captured manifest is an
independent oracle for the loader's weight contract: the expected parameter set
is *derived from config.json* in `hibiki_mlx`, while these fixtures are *read
off the released artifacts*. If the two ever disagree, a test fails.

Usage:
    python tools/capture_weight_manifest.py <bundle-dir> <output.json>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any


def read_header(path: Path) -> dict[str, Any]:
    """Read a safetensors header without touching the tensor data."""
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header: dict[str, Any] = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header


def capture(bundle_dir: Path) -> dict[str, Any]:
    from hibiki_mlx.trust import PINNED_REVISION

    config = json.loads((bundle_dir / "config.json").read_text())
    captured: dict[str, Any] = {
        "source_revision": PINNED_REVISION,
        "config": config,
        "weights": {},
    }
    for role, filename in (("lm", config["moshi_name"]), ("mimi", config["mimi_name"])):
        header = read_header(bundle_dir / filename)
        captured["weights"][role] = {
            name: [entry["dtype"], entry["shape"]] for name, entry in sorted(header.items())
        }
    return captured


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    bundle_dir, output = Path(argv[1]), Path(argv[2])
    captured = capture(bundle_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(captured, indent=1, sort_keys=True) + "\n")
    counts = {role: len(names) for role, names in captured["weights"].items()}
    print(f"wrote {output} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
