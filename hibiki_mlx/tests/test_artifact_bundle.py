"""Behavior of the artifact loader, observed only through `load_artifact_bundle`."""

from __future__ import annotations

import re

import pytest

from hibiki_mlx import (
    ArtifactContentError,
    IncompatibleArtifactError,
    UntrustedArtifactError,
    load_artifact_bundle,
)

from .conftest import BundleBuilder

# Released size of the pinned SentencePiece model. Small enough that a quick test
# can match it exactly and so reach the hash check behind the size check.
TOKENIZER_SIZE = 857314


def test_loads_an_exact_local_bundle(builder: BundleBuilder) -> None:
    """A local bundle matching the release resolves to its four verified artifacts.

    The fixture's parameter names and shapes were captured off the released
    artifacts, so accepting it proves the parameters this package derives from
    `config.json` are the ones the release actually contains.
    """
    root = builder.write()

    bundle = load_artifact_bundle(root, allow_unsafe=True)

    assert bundle.config_path == root / "config.json"
    assert bundle.lm_weights_path == root / "hibiki-mlx-dc2cf5a5@80.safetensors"
    assert bundle.mimi_weights_path == root / "mimi-dbaa9758@125.safetensors"
    assert bundle.tokenizer_path == root / "tokenizer_spm_48k_multi6_2.model"
    assert bundle.spec.n_q == 16
    assert bundle.spec.dep_q == 8


def test_rejects_a_missing_artifact(builder: BundleBuilder) -> None:
    root = builder.write()
    (root / "mimi-dbaa9758@125.safetensors").unlink()

    with pytest.raises(ArtifactContentError, match=re.escape("mimi-dbaa9758@125.safetensors")):
        load_artifact_bundle(root, allow_unsafe=True)


def test_safe_mode_rejects_a_bundle_that_is_not_the_pinned_release(
    builder: BundleBuilder,
) -> None:
    root = builder.write()

    with pytest.raises(UntrustedArtifactError) as raised:
        load_artifact_bundle(root)
    assert "allow_unsafe" in str(raised.value)


def test_safe_mode_reports_every_untrusted_artifact_and_why(builder: BundleBuilder) -> None:
    """A file of the right size still has to hash to the right value."""
    builder.tokenizer_bytes = b"\0" * TOKENIZER_SIZE
    root = builder.write()

    with pytest.raises(UntrustedArtifactError) as raised:
        load_artifact_bundle(root)
    message = str(raised.value)
    assert "SHA-256" in message
    assert "size" in message


def test_waiving_trust_does_not_waive_the_architecture(builder: BundleBuilder) -> None:
    """`allow_unsafe` relaxes revision and hash trust, and nothing else."""
    builder.lm["text_linear.weight"] = ("BF16", [32000, 2048])
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError):
        load_artifact_bundle(root, allow_unsafe=True)
    # Safe mode reports the same unusable architecture, not merely a trust problem.
    with pytest.raises(IncompatibleArtifactError):
        load_artifact_bundle(root)


def test_rejects_a_second_set_of_weights_beside_the_named_ones(
    builder: BundleBuilder,
) -> None:
    """Two candidate weight files make the authoritative one ambiguous."""
    builder.extra_files["hibiki-mlx-dc2cf5a5@80.q4.safetensors"] = b"a quantized variant"
    root = builder.write()

    with pytest.raises(ArtifactContentError, match="q4"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_weights_from_a_revision_the_configuration_does_not_name(
    builder: BundleBuilder,
) -> None:
    """Mixing one revision's config with another's weights must not load."""
    root = builder.write()
    (root / "mimi-dbaa9758@125.safetensors").rename(root / "mimi-0abcdef1@126.safetensors")

    with pytest.raises(ArtifactContentError) as raised:
        load_artifact_bundle(root, allow_unsafe=True)
    message = str(raised.value)
    assert "mimi-dbaa9758@125.safetensors" in message
    assert "mimi-0abcdef1@126.safetensors" in message


def test_ignores_documentation_alongside_the_artifacts(builder: BundleBuilder) -> None:
    """A bundle downloaded from the Hub carries a README and cache metadata."""
    builder.extra_files["README.md"] = b"# Hibiki"
    builder.extra_files[".gitattributes"] = b"*.safetensors filter=lfs"
    root = builder.write()
    (root / ".cache").mkdir()

    assert load_artifact_bundle(root, allow_unsafe=True).spec.dep_q == 8


def test_rejects_lm_weights_missing_a_parameter(builder: BundleBuilder) -> None:
    del builder.lm["transformer.layers.7.self_attn.in_proj.weight"]
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match=re.escape("transformer.layers.7")):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_lm_weights_with_an_unexpected_parameter(builder: BundleBuilder) -> None:
    builder.lm["transformer.layers.0.self_attn.k_bias"] = ("BF16", [2048])
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="k_bias"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_parameter_whose_shape_disagrees_with_the_configuration(
    builder: BundleBuilder,
) -> None:
    """A text head sized for a different vocabulary must not load silently."""
    builder.lm["text_linear.weight"] = ("BF16", [32000, 2048])
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match=re.escape("text_linear.weight")):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_parameter_with_the_wrong_dtype(builder: BundleBuilder) -> None:
    """This scope targets the BF16 release; a repackaged dtype is not equivalent."""
    builder.lm["text_emb.weight"] = ("F16", [48001, 2048])
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="F16"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_mimi_weights_missing_a_parameter(builder: BundleBuilder) -> None:
    del builder.mimi["encoder.model.0.conv.conv.weight"]
    root = builder.write()

    with pytest.raises(
        IncompatibleArtifactError, match=re.escape("encoder.model.0.conv.conv.weight")
    ):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_mimi_weights_with_a_wrong_codebook_size(builder: BundleBuilder) -> None:
    """Codebook cardinality must agree with the configuration's `card`."""
    builder.mimi["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"] = ("F32", [1024, 256])
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="embedding_sum"):
        load_artifact_bundle(root, allow_unsafe=True)
