"""Load the released Hibiki bundle for local MLX inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import sentencepiece

from .download import DEFAULT_ARTIFACT_DIRECTORY
from .models.lm import Lm, LmConfig
from .models.mimi import Mimi, mimi_202407, remap_released_weights
from .text import Tokenizer

CONFIG_FILE = "config.json"


class ModelLoadError(RuntimeError):
    """Raised when a required file in the local model bundle is unavailable."""


@dataclass(frozen=True)
class LoadedModel:
    """The immutable weights of one artifact bundle, ready to translate.

    Streaming state is not here: an inference session asks the codec and the
    generator for their caches when it starts.
    """

    lm_config: LmConfig
    mimi: Mimi
    lm: Lm
    tokenizer: Tokenizer

    def summary(self) -> str:
        return (
            f"loaded {self.mimi.cfg.quantizer_nq} codec codebooks, "
            f"{self.lm.cfg.transformer.num_layers} temporal layers, "
            f"{self.lm_config.target_codebooks} depth slices, "
            f"dtype {self.lm.text_linear.weight.dtype}"
        )


def load_model(
    *,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    dtype: mx.Dtype = mx.bfloat16,
) -> LoadedModel:
    """Load the Mimi and Hibiki weights into MLX modules.

    Both loads are strict: MLX rejects the bundle unless every parameter this
    implementation declares is present with the expected shape, and nothing
    else is. The generator is cast to ``dtype`` before loading so its released
    BF16 weights are never widened. The codec keeps the float32 it ships as.
    """
    config, mimi_path, hibiki_path, tokenizer_path = _bundle_paths(Path(artifact_directory))
    lm_config = LmConfig.from_config_dict(config)

    codebooks = lm_config.target_codebooks
    if lm_config.source_codebooks != codebooks:
        raise ModelLoadError(
            f"the bundle generates {codebooks} audio codebooks but supplies "
            f"{lm_config.source_codebooks}, so one codec cannot serve both streams."
        )

    mimi = Mimi(mimi_202407(codebooks))
    codec_weights, _ = remap_released_weights(
        mx.load(str(mimi_path)),
        codebooks=codebooks,
    )
    mimi.load_weights(list(codec_weights.items()), strict=True)
    mimi.refresh_derived_state()

    lm = Lm(lm_config)
    lm.set_dtype(dtype)
    lm.load_weights(str(hibiki_path), strict=True)

    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    mx.eval(mimi.parameters(), lm.parameters())
    return LoadedModel(
        lm_config=lm_config,
        mimi=mimi,
        lm=lm,
        tokenizer=tokenizer,
    )


def _bundle_paths(directory: Path) -> tuple[dict[str, object], Path, Path, Path]:
    config = _read_config(directory)
    return (
        config,
        _artifact_path(directory, config, "mimi_name"),
        _artifact_path(directory, config, "moshi_name"),
        _artifact_path(directory, config, "tokenizer_name"),
    )


def _read_config(directory: Path) -> dict[str, object]:
    path = directory / CONFIG_FILE
    try:
        config = json.loads(path.read_text())
    except OSError as error:
        raise ModelLoadError(
            f"{path} could not be read: {error}. Download the artifacts first."
        ) from error
    except json.JSONDecodeError as error:
        raise ModelLoadError(f"{path} is not valid JSON: {error}.") from error
    if not isinstance(config, dict):
        raise ModelLoadError(f"{path} does not contain a configuration object.")
    return config


def _artifact_path(directory: Path, config: dict[str, object], key: str) -> Path:
    name = config.get(key)
    if not isinstance(name, str) or not name:
        raise ModelLoadError(f"{directory / CONFIG_FILE} does not name its {key} artifact.")
    path = directory / name
    if not path.is_file():
        raise ModelLoadError(
            f"{path} is missing, but the bundle configuration names it as {key}."
        )
    return path
