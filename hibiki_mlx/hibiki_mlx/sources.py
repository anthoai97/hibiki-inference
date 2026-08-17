"""Turning a caller's `source` into a directory holding one bundle.

Two kinds of source are supported: a prepared local directory, and a Hugging Face
repository at an explicit revision. Both end as a directory the rest of the
loader can verify, so verification never depends on where the bundle came from.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import ArtifactResolutionError
from .spec import CONFIG_FILENAME
from .trust import PINNED_REVISION

# `namespace/name`, the only shape a Hugging Face model id takes here.
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def is_local_source(source: str | Path) -> bool:
    """Whether `source` names a local bundle rather than a repository.

    An existing directory always wins over a repository id, so a local path that
    happens to look like `namespace/name` is never fetched from the network. Pass
    a `Path` to require the local interpretation.
    """
    return isinstance(source, Path) or Path(source).exists()


def resolve_source(
    source: str | Path,
    *,
    revision: str | None,
    offline: bool,
    allow_unsafe: bool,
) -> Path:
    """Return the directory containing the bundle named by `source`."""
    path = Path(source)
    if is_local_source(source):
        if revision is not None:
            raise ArtifactResolutionError(
                f"revision={revision!r} was given for the local bundle {path}, but "
                f"a directory has no revision to select. Drop the revision, or "
                f"pass a Hugging Face repository id to choose one."
            )
        return _resolve_local(path)

    text = str(source)
    if not _REPO_ID.match(text):
        raise ArtifactResolutionError(
            f"{text!r} is neither an existing directory nor a Hugging Face "
            f"repository id of the form 'namespace/name'."
        )
    return _resolve_hub(text, revision=revision, offline=offline, allow_unsafe=allow_unsafe)


def _resolve_local(path: Path) -> Path:
    if not path.exists():
        raise ArtifactResolutionError(f"No artifact bundle at {path}: the path does not exist.")
    if not path.is_dir():
        raise ArtifactResolutionError(
            f"No artifact bundle at {path}: a bundle is a directory of four "
            f"artifacts, not a single file."
        )
    return path


def _resolve_hub(
    repo_id: str,
    *,
    revision: str | None,
    offline: bool,
    allow_unsafe: bool,
) -> Path:
    """Fetch or locate a Hub revision, downloading only the four artifacts."""
    if revision is None:
        # Safe mode has exactly one revision, so defaulting to it is unambiguous.
        # An unpinned revision tracks a moving branch, and so is reachable only by
        # opting out of trust.
        revision = None if allow_unsafe else PINNED_REVISION
    described = f"revision {revision}" if revision else "its default revision"

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        downloaded = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=offline,
            allow_patterns=[CONFIG_FILENAME, "*.safetensors", "*.model"],
        )
    except LocalEntryNotFoundError as error:
        if offline:
            raise ArtifactResolutionError(
                f"{repo_id} at {described} is not in the local Hugging "
                f"Face cache, and offline mode makes no network request. Prepare "
                f"the bundle first, or load it from a local directory."
            ) from error
        raise ArtifactResolutionError(
            f"{repo_id} at {described} could not be resolved: {error}."
        ) from error
    except HfHubHTTPError as error:
        raise ArtifactResolutionError(
            f"{repo_id} at {described} could not be fetched from the Hugging Face Hub: {error}."
        ) from error
    except OSError as error:
        raise ArtifactResolutionError(
            f"{repo_id} at {described} could not be read from the local "
            f"Hugging Face cache: {error}."
        ) from error

    return Path(downloaded)
