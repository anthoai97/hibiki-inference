"""What this package trusts as the released artifact bundle.

Safe mode accepts one revision and one set of bytes. Everything here is about
identity -- "are these the artifacts this implementation was verified against?"
-- and is the only thing `allow_unsafe=True` waives. Whether a bundle is
*structurally usable* is decided elsewhere, and is never waived.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .errors import UntrustedArtifactError

PINNED_REPO_ID = "kyutai/hibiki-1b-mlx-bf16"
PINNED_REVISION = "b3d6291f3dcf7954e1a502e4d66f32e3556f17ae"

# Read in 8 MiB blocks: large enough to keep hashing 3.6 GB I/O-bound, small
# enough not to matter against unified memory.
_HASH_BLOCK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TrustedArtifact:
    size: int
    sha256: str


# Sizes and digests of the pinned revision, recorded from the release.
TRUSTED_ARTIFACTS: dict[str, TrustedArtifact] = {
    "config.json": TrustedArtifact(
        size=1364,
        sha256="fa60fd34d98db63cf7673766c3053b56370aa0385c5e3e8d2922bed163bb3758",
    ),
    "hibiki-mlx-dc2cf5a5@80.safetensors": TrustedArtifact(
        size=3600043224,
        sha256="2d1baa58b2003aef24a034cdec5bc8c6b4c6d14d0d50e530c42708e62e0b30d9",
    ),
    "mimi-dbaa9758@125.safetensors": TrustedArtifact(
        size=384644900,
        sha256="31c14cf365353131094e8248150c6fe58e8642cf91899c50d9e450f861630e55",
    ),
    "tokenizer_spm_48k_multi6_2.model": TrustedArtifact(
        size=857314,
        sha256="c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697",
    ),
}


def verify_revision(revision: str | None) -> None:
    """Reject any revision other than the pinned one."""
    if revision != PINNED_REVISION:
        named = f"revision {revision!r}" if revision else "an unpinned revision"
        raise UntrustedArtifactError(
            f"Safe mode loads only {PINNED_REPO_ID} at revision {PINNED_REVISION}, "
            f"but {named} was requested. Pass the pinned revision, or pass "
            f"allow_unsafe=True to accept another one -- which waives revision "
            f"and hash trust only, never file, configuration, or shape checks."
        )


def verify_artifact_bytes(paths: dict[str, Path]) -> None:
    """Reject any artifact whose size or digest is not the released one.

    Sizes are checked before digests so a bundle that is obviously not the
    release fails without reading gigabytes.
    """
    problems: list[str] = []
    for name, path in sorted(paths.items()):
        trusted = TRUSTED_ARTIFACTS.get(name)
        if trusted is None:
            problems.append(f"{name} is not an artifact of the pinned revision")
            continue
        actual_size = path.stat().st_size
        if actual_size != trusted.size:
            problems.append(
                f"{name} has size {actual_size} but the release is {trusted.size} bytes"
            )
            continue
        digest = _sha256(path)
        if digest != trusted.sha256:
            problems.append(f"{name} has SHA-256 {digest} but the release is {trusted.sha256}")

    if problems:
        raise UntrustedArtifactError(
            f"Safe mode did not recognize {len(problems)} artifact(s) as the "
            f"pinned release: {'; '.join(problems)}. Pass allow_unsafe=True to "
            f"load them anyway -- that waives revision and hash trust only, "
            f"never file, configuration, or shape checks."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
