"""Start inference by checking and then loading the artifact bundle's weights.

The Mimi codec comes first and the Hibiki generator second, matching the order
the frame loop uses them: source PCM is encoded into audio tokens before the
generator runs.

``start()`` only reads the safetensors headers, so the bundle's four gigabytes
are checked in milliseconds and nothing is allocated before the weights are
known to be readable. ``load_model()`` goes on to build the local MLX modules
and strict-load both files into them.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path

import mlx.core as mx
import sentencepiece

from .download import DEFAULT_ARTIFACT_DIRECTORY
from .models.lm import Lm, LmConfig
from .models.mimi import Mimi, mimi_202407, remap_released_weights
from .text import Tokenizer

CONFIG_FILE = "config.json"

# A header larger than this belongs to no release this project targets, so it is
# rejected before the bytes are read into memory.
_MAX_HEADER_BYTES = 64 * 1024 * 1024

TensorManifest = dict[str, tuple[str, tuple[int, ...]]]


class WeightCheckError(RuntimeError):
    """Raised when an artifact cannot be read or does not hold usable weights."""


@dataclass(frozen=True)
class WeightCheck:
    """What one checked safetensors artifact contains."""

    role: str
    path: Path
    tensors: TensorManifest

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def parameter_count(self) -> int:
        return sum(prod(shape) for _, shape in self.tensors.values())

    @property
    def dtypes(self) -> tuple[str, ...]:
        return tuple(sorted({dtype for dtype, _ in self.tensors.values()}))

    @property
    def prefixes(self) -> tuple[str, ...]:
        """The top-level parameter groups, which name the modules to build."""
        return tuple(sorted({name.split(".", 1)[0] for name in self.tensors}))

    def summary(self) -> str:
        return (
            f"{self.role}: {self.path.name}\n"
            f"  tensors    {self.tensor_count}\n"
            f"  parameters {self.parameter_count:,}\n"
            f"  dtypes     {', '.join(self.dtypes)}\n"
            f"  groups     {', '.join(self.prefixes)}"
        )


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
    mimi_check: WeightCheck
    hibiki_check: WeightCheck
    unused_codec_weights: tuple[str, ...]

    def summary(self) -> str:
        codebooks = self.mimi.cfg.quantizer_nq
        return (
            f"{self.mimi_check.summary()}\n"
            f"  loaded     {codebooks} codebooks, "
            f"{len(self.unused_codec_weights)} unused codebook tensors dropped\n"
            f"{self.hibiki_check.summary()}\n"
            f"  loaded     {self.lm.cfg.transformer.num_layers} temporal layers, "
            f"{self.lm_config.target_codebooks} depth slices, "
            f"dtype {self.lm.text_linear.weight.dtype}"
        )


def start(
    *,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> tuple[WeightCheck, WeightCheck]:
    """Check the Mimi weights, then the Hibiki weights, and return both results.

    The bundle's own ``config.json`` names the two safetensors files, so a
    configuration and weights from different revisions cannot be paired up
    silently.
    """
    _, mimi, hibiki = _check_bundle(Path(artifact_directory))
    return mimi, hibiki


def load_model(
    *,
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
    dtype: mx.Dtype = mx.bfloat16,
) -> LoadedModel:
    """Check the bundle, then load the Mimi and Hibiki weights into MLX modules.

    Both loads are strict: MLX rejects the bundle unless every parameter this
    implementation declares is present with the expected shape, and nothing
    else is. The generator is cast to ``dtype`` before loading so its released
    BF16 weights are never widened. The codec keeps the float32 it ships as.
    """
    config, mimi_check, hibiki_check = _check_bundle(Path(artifact_directory))
    lm_config = LmConfig.from_config_dict(config)

    codebooks = lm_config.target_codebooks
    if lm_config.source_codebooks != codebooks:
        raise WeightCheckError(
            f"the bundle generates {codebooks} audio codebooks but supplies "
            f"{lm_config.source_codebooks}, so one codec cannot serve both streams."
        )

    mimi = Mimi(mimi_202407(codebooks))
    codec_weights, unused = remap_released_weights(
        mx.load(str(mimi_check.path)),
        codebooks=codebooks,
    )
    mimi.load_weights(list(codec_weights.items()), strict=True)
    mimi.refresh_derived_state()

    lm = Lm(lm_config)
    lm.set_dtype(dtype)
    lm.load_weights(str(hibiki_check.path), strict=True)

    tokenizer = sentencepiece.SentencePieceProcessor(
        str(_artifact_path(Path(artifact_directory), config, "tokenizer_name"))
    )

    mx.eval(mimi.parameters(), lm.parameters())
    return LoadedModel(
        lm_config=lm_config,
        mimi=mimi,
        lm=lm,
        tokenizer=tokenizer,
        mimi_check=mimi_check,
        hibiki_check=hibiki_check,
        unused_codec_weights=unused,
    )


def _check_bundle(directory: Path) -> tuple[dict[str, object], WeightCheck, WeightCheck]:
    config = _read_config(directory)
    mimi = check_weights(_artifact_path(directory, config, "mimi_name"), role="Mimi codec")
    hibiki = check_weights(_artifact_path(directory, config, "moshi_name"), role="Hibiki generator")
    return config, mimi, hibiki


def check_weights(path: Path, *, role: str) -> WeightCheck:
    """Read one safetensors artifact's parameter names, dtypes, and shapes."""
    tensors = read_tensor_manifest(path)
    if not tensors:
        raise WeightCheckError(f"{path} holds no tensors, so it carries no {role} weights.")
    return WeightCheck(role=role, path=path, tensors=tensors)


def read_tensor_manifest(path: Path) -> TensorManifest:
    """Return every tensor's dtype and shape, keyed by parameter name."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise WeightCheckError(
                    f"{path} is too small to be a safetensors file: it has no header length."
                )
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length == 0 or header_length > _MAX_HEADER_BYTES:
                raise WeightCheckError(
                    f"{path} declares an implausible safetensors header of "
                    f"{header_length} bytes, so it is not a readable artifact."
                )
            encoded = handle.read(header_length)
    except OSError as error:
        raise WeightCheckError(f"{path} could not be read: {error}.") from error

    if len(encoded) < header_length:
        raise WeightCheckError(
            f"{path} is truncated: its header claims {header_length} bytes but only "
            f"{len(encoded)} are present."
        )
    try:
        header = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise WeightCheckError(
            f"{path} has a safetensors header that is not valid JSON: {error}."
        ) from error
    if not isinstance(header, dict):
        raise WeightCheckError(f"{path} has a safetensors header that is not an object.")

    header.pop("__metadata__", None)
    manifest: TensorManifest = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or "dtype" not in entry or "shape" not in entry:
            raise WeightCheckError(f"{path} describes tensor {name!r} without a dtype and shape.")
        manifest[name] = (str(entry["dtype"]), tuple(int(extent) for extent in entry["shape"]))
    return manifest


def _read_config(directory: Path) -> dict[str, object]:
    path = directory / CONFIG_FILE
    try:
        config = json.loads(path.read_text())
    except OSError as error:
        raise WeightCheckError(
            f"{path} could not be read: {error}. Download the artifacts first."
        ) from error
    except json.JSONDecodeError as error:
        raise WeightCheckError(f"{path} is not valid JSON: {error}.") from error
    if not isinstance(config, dict):
        raise WeightCheckError(f"{path} does not contain a configuration object.")
    return config


def _artifact_path(directory: Path, config: dict[str, object], key: str) -> Path:
    name = config.get(key)
    if not isinstance(name, str) or not name:
        raise WeightCheckError(f"{directory / CONFIG_FILE} does not name its {key} artifact.")
    path = directory / name
    if not path.is_file():
        raise WeightCheckError(
            f"{path} is missing, but the bundle configuration names it as {key}."
        )
    return path


