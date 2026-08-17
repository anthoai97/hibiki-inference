"""Configuration drift that must be rejected before anything is allocated.

Each case here is a value some later stage silently depends on. A wrong
cardinality or layer count would surface as a shape error, but a wrong delay
schedule or conditioner ordering would surface as plausible, wrong audio -- so
the configuration is checked against this implementation up front.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from hibiki_mlx import ArtifactContentError, IncompatibleArtifactError, load_artifact_bundle

from .conftest import BundleBuilder

# Field, replacement value, and the text the resulting error must name.
DRIFT_CASES: list[tuple[str, Any, str]] = [
    # Cardinalities.
    ("card", 1024, "card"),
    ("text_card", 32000, "text_card"),
    ("dim", 1024, "dim"),
    # Codebook split: 16 temporal audio streams are 8 target plus 8 source.
    ("n_q", 8, "n_q"),
    ("dep_q", 16, "dep_q"),
    # Attention context, which the session's cache is sized around.
    ("context", 4096, "context"),
    # Layer counts.
    ("num_layers", 24, "num_layers"),
    ("num_heads", 8, "num_heads"),
    ("depformer_num_layers", 4, "depformer_num_layers"),
    ("depformer_dim", 512, "depformer_dim"),
    # Structural choices this implementation does not implement alternatives for.
    ("gating", "gelu", "gating"),
    ("norm", "layer_norm", "norm"),
    ("positional_embedding", "sin", "positional_embedding"),
    ("depformer_weights_per_step", False, "depformer_weights_per_step"),
    ("existing_text_padding_id", 0, "existing_text_padding_id"),
]


@pytest.mark.parametrize(("field", "value", "expected"), DRIFT_CASES)
def test_rejects_configuration_drift(
    builder: BundleBuilder, field: str, value: Any, expected: str
) -> None:
    builder.config[field] = value
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError) as raised:
        load_artifact_bundle(root, allow_unsafe=True)
    message = str(raised.value)
    assert expected in message
    # The configuration itself must be blamed, not a shape it happens to change.
    assert "config.json" in message


def test_rejects_a_delay_schedule_of_the_wrong_length(builder: BundleBuilder) -> None:
    builder.config["delays"] = builder.config["delays"][:-1]
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="delays"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_delay_schedule_in_the_wrong_order(builder: BundleBuilder) -> None:
    """Delaying the semantic codebook, or not delaying an acoustic one, changes timing."""
    delays = list(builder.config["delays"])
    delays[1], delays[2] = delays[2], delays[1]
    builder.config["delays"] = delays
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="delays"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_unexpected_conditioner_labels(builder: BundleBuilder) -> None:
    """Label ordering decides which embedding row voice conditioning selects."""
    lut = builder.config["conditioners"]["description"]["lut"]
    lut["possible_values"] = ["bad", "very_bad", "neutral", "good", "very_good"]
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="labels"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_an_unknown_conditioner(builder: BundleBuilder) -> None:
    builder.config["conditioners"]["speaker"] = {"type": "lut"}
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="description"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_configuration_missing_a_required_key(builder: BundleBuilder) -> None:
    del builder.config["depformer_dim_feedforward"]
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="depformer_dim_feedforward"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_bundle_without_a_configuration(builder: BundleBuilder) -> None:
    root = builder.write()
    (root / "config.json").unlink()

    with pytest.raises(ArtifactContentError, match=re.escape("config.json")):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_configuration_that_is_not_valid_json(builder: BundleBuilder) -> None:
    root = builder.write()
    (root / "config.json").write_text("{ not json")

    with pytest.raises(ArtifactContentError, match="valid JSON"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_configuration_that_is_not_valid_text(builder: BundleBuilder) -> None:
    root = builder.write()
    (root / "config.json").write_bytes(b"\xff\xfe\x00 not utf-8")

    with pytest.raises(ArtifactContentError):
        load_artifact_bundle(root, allow_unsafe=True)


# JSON that parses but is not shaped like a configuration. Each of these once
# escaped as a bare TypeError instead of a public error.
MALFORMED_CONFIGS: list[tuple[str, str]] = [
    ("a bare number", "5"),
    ("a list", '["dim", 2048]'),
    ("a string", '"config"'),
    ("null", "null"),
]


@pytest.mark.parametrize(("described", "content"), MALFORMED_CONFIGS)
def test_rejects_a_configuration_that_is_not_an_object(
    builder: BundleBuilder, described: str, content: str
) -> None:
    root = builder.write()
    (root / "config.json").write_text(content)

    with pytest.raises(IncompatibleArtifactError, match="object is required"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_conditioners_that_are_not_an_object(builder: BundleBuilder) -> None:
    builder.config["conditioners"] = ["description"]
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="object is required"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_conditioner_without_a_lookup_table(builder: BundleBuilder) -> None:
    builder.config["conditioners"]["description"] = {"type": "lut"}
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="object is required"):
        load_artifact_bundle(root, allow_unsafe=True)


def test_rejects_a_fuser_that_is_not_an_object(builder: BundleBuilder) -> None:
    builder.config["fuser"] = "sum"
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match="object is required"):
        load_artifact_bundle(root, allow_unsafe=True)


@pytest.mark.parametrize(("field", "value"), [("dim", 32), ("n_bins", 64)])
def test_rejects_conditioner_geometry_drift(builder: BundleBuilder, field: str, value: int) -> None:
    """The conditioner's embedding shapes derive from these, so they must be pinned."""
    builder.config["conditioners"]["description"]["lut"][field] = value
    root = builder.write()

    with pytest.raises(IncompatibleArtifactError, match=f"lut.{field}"):
        load_artifact_bundle(root, allow_unsafe=True)
