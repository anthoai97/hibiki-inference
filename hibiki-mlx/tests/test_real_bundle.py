"""Verification against the real released bundle.

These tests are opt-in (`pytest -m model_backed`) because they need the ~4 GB
artifact bundle on disk. They are what proves the quick tests are checking the
right thing: the synthetic fixtures replay a captured manifest, while these load
the release itself in safe mode -- exact revision sizes, exact SHA-256 digests,
and every parameter of 3.98 GB of real weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hibiki_mlx import UntrustedArtifactError, load_artifact_bundle

pytestmark = pytest.mark.model_backed

# Where the repository's own download lands, used when the variable is unset.
DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "artifacts" / "hibiki-1b-mlx-bf16"


@pytest.fixture(scope="module")
def real_bundle_dir() -> Path:
    configured = os.environ.get("HIBIKI_MLX_TEST_BUNDLE")
    path = Path(configured) if configured else DEFAULT_BUNDLE
    if not (path / "config.json").is_file():
        pytest.skip(
            f"No artifact bundle at {path}. Set HIBIKI_MLX_TEST_BUNDLE to a "
            f"prepared bundle to run model-backed tests."
        )
    return path


def test_safe_mode_accepts_the_released_bundle(real_bundle_dir: Path) -> None:
    """The release passes every check with no trust waived."""
    bundle = load_artifact_bundle(real_bundle_dir)

    assert bundle.lm_weights_path.name == "hibiki-mlx-dc2cf5a5@80.safetensors"
    assert bundle.mimi_weights_path.name == "mimi-dbaa9758@125.safetensors"
    assert bundle.tokenizer_path.name == "tokenizer_spm_48k_multi6_2.model"

    spec = bundle.spec
    assert (spec.n_q, spec.dep_q, spec.card, spec.text_card) == (16, 8, 2048, 48000)
    assert (spec.dim, spec.num_layers, spec.num_heads, spec.context) == (2048, 16, 16, 500)
    assert spec.depformer_num_layers == 6
    assert spec.gated_width == 5632
    assert spec.depformer_gated_width == 2816
    assert spec.delays == (0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2)


def test_safe_mode_accepts_the_pinned_hub_revision() -> None:
    """The remote path, from repository id to a verified bundle.

    Gated separately because it downloads about 4 GB into the Hugging Face cache
    even when a local bundle is already present:
    `HIBIKI_MLX_TEST_HUB=1 pytest -m model_backed`.
    """
    if os.environ.get("HIBIKI_MLX_TEST_HUB") != "1":
        pytest.skip("Set HIBIKI_MLX_TEST_HUB=1 to download the pinned revision from the Hub.")

    bundle = load_artifact_bundle("kyutai/hibiki-1b-mlx-bf16")

    assert bundle.spec.dep_q == 8
    assert bundle.lm_weights_path.is_file()


def test_a_tampered_release_is_not_trusted(real_bundle_dir: Path, tmp_path: Path) -> None:
    """Same names and shapes, one changed byte: safe mode must still refuse it."""
    copy = tmp_path / "bundle"
    copy.mkdir()
    for artifact in real_bundle_dir.iterdir():
        if artifact.is_file():
            (copy / artifact.name).write_bytes(artifact.read_bytes())

    tokenizer = copy / "tokenizer_spm_48k_multi6_2.model"
    payload = bytearray(tokenizer.read_bytes())
    payload[-1] ^= 0xFF
    tokenizer.write_bytes(payload)

    with pytest.raises(UntrustedArtifactError, match="SHA-256"):
        load_artifact_bundle(copy)

    # The tampered bundle is still structurally sound, so waiving trust loads it.
    assert load_artifact_bundle(copy, allow_unsafe=True).spec.dep_q == 8
