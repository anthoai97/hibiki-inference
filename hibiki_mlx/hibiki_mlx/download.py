"""Download the pinned Hibiki artifacts into the repository's artifact directory."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download as _hf_hub_download

MODEL_REPOSITORY = "kyutai/hibiki-1b-mlx-bf16"
MODEL_REVISION = "b3d6291f3dcf7954e1a502e4d66f32e3556f17ae"
MODEL_FILES = (
    "config.json",
    "hibiki-mlx-dc2cf5a5@80.safetensors",
    "mimi-dbaa9758@125.safetensors",
    "tokenizer_spm_48k_multi6_2.model",
)
DEFAULT_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "artifacts" / MODEL_REPOSITORY.rsplit("/", 1)[1]
)


def download_model(
    *,
    destination: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    revision: str = MODEL_REVISION,
) -> Path:
    """Download the required model files and return their local directory.

    Downloads are pinned to the released Hibiki MLX revision by default. Passing
    a different revision is useful only when intentionally testing another model
    revision. The destination defaults to ``artifacts/hibiki-1b-mlx-bf16`` at
    the repository root, which is excluded from version control.
    """
    artifact_directory = Path(destination)
    artifact_directory.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        _hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=filename,
            revision=revision,
            local_dir=artifact_directory,
        )

    return artifact_directory
