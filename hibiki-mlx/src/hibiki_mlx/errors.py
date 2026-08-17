"""Public error types.

Every failure a caller can encounter is one of these. They are deliberately
distinguishable rather than one catch-all, because the remedies differ: a
resolution failure means "point me somewhere else", an untrusted bundle means
"you are not loading the release", and an incompatible bundle means "these
artifacts cannot drive this implementation".
"""

from __future__ import annotations


class HibikiError(Exception):
    """Base class for every error this package raises."""


class ArtifactError(HibikiError):
    """An artifact bundle could not be resolved or verified."""


class ArtifactResolutionError(ArtifactError):
    """The bundle could not be located.

    A local directory does not exist, a Hugging Face repository id is malformed,
    or `offline=True` was set and the bundle is not already cached.
    """


class ArtifactContentError(ArtifactError):
    """The bundle was found but its contents are wrong.

    A required artifact is missing, an unexpected file sits alongside the
    required ones, or `config.json` names files other than the ones it was
    loaded with -- which is how artifacts from mixed revisions are caught.
    """


class UntrustedArtifactError(ArtifactError):
    """Safe mode declined to trust the bundle.

    The revision is not the pinned one, or a file's size or SHA-256 does not
    match its built-in value. This is the only error `allow_unsafe=True`
    suppresses; it never relaxes file, configuration, name, or shape checks.
    """


class IncompatibleArtifactError(ArtifactError):
    """The artifacts do not match this implementation.

    A configuration invariant, a parameter name, a dtype, or a tensor shape
    differs from the architecture this package implements.
    """
