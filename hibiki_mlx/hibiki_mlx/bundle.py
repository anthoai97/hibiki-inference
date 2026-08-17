"""Resolving and verifying an artifact bundle.

An artifact bundle is the configuration, the Hibiki weights, the Mimi weights,
and the SentencePiece model from one model revision, treated as one verified
unit. `load_artifact_bundle` is the only way to obtain one, so no later stage can
run against artifacts that were never checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ArtifactContentError
from .sources import is_local_source, resolve_source
from .spec import CONFIG_FILENAME, ModelSpec, parse_model_spec
from .trust import verify_artifact_bytes, verify_revision
from .weights import expected_lm_parameters, expected_mimi_parameters, verify_weights


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """Four artifacts from one revision, verified together."""

    config_path: Path
    lm_weights_path: Path
    mimi_weights_path: Path
    tokenizer_path: Path
    spec: ModelSpec


def load_artifact_bundle(
    source: str | Path,
    *,
    revision: str | None = None,
    offline: bool = False,
    allow_unsafe: bool = False,
) -> ArtifactBundle:
    """Resolve and verify an artifact bundle.

    `source` is either a prepared local bundle directory or a Hugging Face
    repository id, in which case `revision` selects the revision and defaults to
    the pinned one. `offline=True` makes no network request.

    Verification covers the presence of every required artifact, that they all
    belong to the revision the configuration describes, every configuration
    invariant this implementation depends on, and every parameter name, dtype,
    and shape in both sets of weights. Only safetensors headers are read, so
    nothing is allocated and no weight values are touched.

    In safe mode the revision, and each artifact's size and SHA-256, must be the
    released ones. `allow_unsafe=True` waives that identity check and nothing
    else, so an untrusted bundle is still rejected unless it structurally matches
    this implementation.

    Raises:
        ArtifactResolutionError: the bundle could not be located.
        ArtifactContentError: the bundle's contents are missing, mixed, or unreadable.
        UntrustedArtifactError: safe mode did not recognize it as the release.
        IncompatibleArtifactError: it does not match this implementation.
    """
    if not allow_unsafe and revision is not None and not is_local_source(source):
        # Check trust before resolving, so safe mode never fetches a revision it
        # was always going to reject.
        verify_revision(revision)

    root = resolve_source(source, revision=revision, offline=offline, allow_unsafe=allow_unsafe)
    spec = _load_spec(root)
    artifacts = _locate_artifacts(root, spec)
    lm_path = artifacts[spec.lm_weights_name]
    mimi_path = artifacts[spec.mimi_weights_name]

    # Structural checks come first: `allow_unsafe` can never make an
    # architecturally incompatible bundle usable, and reporting the fundamental
    # problem is more useful than reporting that it is also unrecognized. It
    # also avoids hashing gigabytes of a bundle that could never have loaded.
    verify_weights(
        lm_path,
        expected_lm_parameters(spec),
        role="language model",
        described_by=CONFIG_FILENAME,
    )
    verify_weights(
        mimi_path,
        expected_mimi_parameters(spec),
        role="codec",
        described_by="this implementation",
    )

    if not allow_unsafe:
        verify_artifact_bytes(artifacts)

    return ArtifactBundle(
        config_path=artifacts[CONFIG_FILENAME],
        lm_weights_path=lm_path,
        mimi_weights_path=mimi_path,
        tokenizer_path=artifacts[spec.tokenizer_name],
        spec=spec,
    )


def _load_spec(root: Path) -> ModelSpec:
    config_path = root / CONFIG_FILENAME
    if not config_path.is_file():
        raise ArtifactContentError(
            f"The bundle at {root} has no {CONFIG_FILENAME}. A bundle needs the "
            f"configuration that names and describes its weights."
        )
    try:
        raw = json.loads(config_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ArtifactContentError(f"{config_path} is not valid JSON: {error}.") from error
    except UnicodeDecodeError as error:
        raise ArtifactContentError(
            f"{config_path} is not valid UTF-8 text, so it cannot be a configuration: {error}."
        ) from error
    except OSError as error:
        raise ArtifactContentError(f"{config_path} could not be read: {error}.") from error
    return parse_model_spec(raw)


# Suffixes that mark a file as a model artifact rather than documentation.
_ARTIFACT_SUFFIXES = (".safetensors", ".model")


def _locate_artifacts(root: Path, spec: ModelSpec) -> dict[str, Path]:
    """Locate the bundle's four artifacts by name, and reject anything else.

    The configuration names its own siblings, which makes it the authority on
    what belongs in the bundle. Requiring an exact match on model artifacts is
    what catches a bundle assembled from more than one revision: the named file
    is absent and an unnamed one is sitting in its place. Documentation and cache
    metadata that ship with a Hub download are ignored.
    """
    names = (spec.lm_weights_name, spec.mimi_weights_name, spec.tokenizer_name)

    missing = [name for name in names if not (root / name).is_file()]
    present = {
        entry.name
        for entry in root.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.name.endswith(_ARTIFACT_SUFFIXES)
    }
    unnamed = sorted(present - set(names))

    if missing or unnamed:
        problems = []
        if missing:
            problems.append(
                f"{CONFIG_FILENAME} names {len(missing)} artifact(s) that are "
                f"absent: {', '.join(missing)}"
            )
        if unnamed:
            problems.append(
                f"{len(unnamed)} model artifact(s) are present that "
                f"{CONFIG_FILENAME} does not name: {', '.join(unnamed)}"
            )
        raise ArtifactContentError(
            f"The bundle at {root} is not one consistent revision -- "
            f"{'; '.join(problems)}. Every artifact must come from the revision "
            f"its {CONFIG_FILENAME} describes."
        )

    return {name: root / name for name in (CONFIG_FILENAME, *names)}
