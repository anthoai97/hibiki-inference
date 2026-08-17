"""Choosing where a bundle comes from: a local directory or a pinned Hub revision."""

from __future__ import annotations

from pathlib import Path

import pytest

from hibiki_mlx import (
    ArtifactResolutionError,
    UntrustedArtifactError,
    load_artifact_bundle,
)
from hibiki_mlx.trust import PINNED_REVISION

from .conftest import BundleBuilder

UNCACHED_REPO = "kyutai/hibiki-1b-mlx-bf16-does-not-exist"


def test_rejects_a_path_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(ArtifactResolutionError, match="does not exist"):
        load_artifact_bundle(tmp_path / "absent", allow_unsafe=True)


def test_rejects_a_file_where_a_bundle_directory_belongs(tmp_path: Path) -> None:
    weights = tmp_path / "hibiki.safetensors"
    weights.write_bytes(b"")

    with pytest.raises(ArtifactResolutionError, match="directory"):
        load_artifact_bundle(weights, allow_unsafe=True)


def test_rejects_a_string_that_is_neither_a_directory_nor_a_repository_id() -> None:
    with pytest.raises(ArtifactResolutionError, match="repository"):
        load_artifact_bundle("hibiki-1b-mlx-bf16", allow_unsafe=True)


def test_an_existing_directory_is_never_treated_as_a_repository_id(
    builder: BundleBuilder,
) -> None:
    """A local bundle resolves locally even when its name looks like a repo id."""
    root = builder.write()

    bundle = load_artifact_bundle(str(root), allow_unsafe=True)

    assert bundle.config_path.parent == root


def test_rejects_a_revision_for_a_local_bundle(builder: BundleBuilder) -> None:
    """A directory has no revision to select, so naming one is a mistake."""
    root = builder.write()

    with pytest.raises(ArtifactResolutionError, match="revision"):
        load_artifact_bundle(root, revision=PINNED_REVISION, allow_unsafe=True)


def test_safe_mode_rejects_an_unpinned_remote_revision() -> None:
    """Safe mode declines before any network request is made."""
    with pytest.raises(UntrustedArtifactError, match=PINNED_REVISION):
        load_artifact_bundle("kyutai/hibiki-1b-mlx-bf16", revision="main")


def test_offline_mode_loads_a_prepared_local_bundle(builder: BundleBuilder) -> None:
    """The offline success path: a prepared bundle needs no network at all."""
    root = builder.write()

    bundle = load_artifact_bundle(root, offline=True, allow_unsafe=True)

    assert bundle.spec.dep_q == 8


def test_an_unpinned_remote_revision_needs_the_unsafe_opt_in() -> None:
    """Safe mode substitutes the pinned revision rather than tracking a branch."""
    with pytest.raises(ArtifactResolutionError) as raised:
        load_artifact_bundle("kyutai/nonexistent-hibiki-repo", offline=True)
    # Safe mode reached the Hub with the pin supplied for it.
    assert PINNED_REVISION in str(raised.value)


def test_waiving_trust_allows_an_unpinned_remote_revision() -> None:
    """With trust waived, an unnamed revision resolves to the repository default."""
    with pytest.raises(ArtifactResolutionError) as raised:
        load_artifact_bundle("kyutai/nonexistent-hibiki-repo", offline=True, allow_unsafe=True)
    message = str(raised.value)
    assert "default revision" in message
    assert PINNED_REVISION not in message


def test_offline_mode_fails_instead_of_fetching_an_uncached_bundle() -> None:
    with pytest.raises(ArtifactResolutionError) as raised:
        load_artifact_bundle(
            UNCACHED_REPO, revision=PINNED_REVISION, offline=True, allow_unsafe=True
        )
    assert "offline" in str(raised.value).lower()
