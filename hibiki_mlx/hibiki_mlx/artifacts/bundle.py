"""The on-disk contract for quantized Hibiki artifact bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx

from ..models.lm import Lm, LmConfig
from .quantization import QUANTIZATION_CONFIG_KEY, QuantizationSpec, quantize_linear_layers


def convert_bundle(source: Path, destination: Path, spec: QuantizationSpec) -> Path:
    """Create a self-contained Q8/Q4 bundle without mutating its BF16 source."""
    source = source.resolve()
    destination = destination.resolve()
    config = _read_config(source)
    source_lm, source_mimi, source_tokenizer = _source_files(source, config)
    _ensure_new_destination(destination)
    destination.mkdir(parents=True)

    try:
        lm_config = LmConfig.from_config_dict(config)
        lm = Lm(lm_config)
        lm.set_dtype(mx.bfloat16)
        lm.load_weights(str(source_lm), strict=True)
        quantize_linear_layers(lm, spec)
        mx.eval(lm.parameters())

        quantized_name = f"{source_lm.stem}.q{spec.bits}.safetensors"
        quantized_path = destination / quantized_name
        lm.save_weights(str(quantized_path))

        shutil.copy2(source_mimi, destination / source_mimi.name)
        shutil.copy2(source_tokenizer, destination / source_tokenizer.name)
        output_config = dict(config)
        output_config["moshi_name"] = quantized_name
        output_config[QUANTIZATION_CONFIG_KEY] = spec.as_config()
        (destination / "config.json").write_text(json.dumps(output_config, indent=2) + "\n")
        (destination / "quantization.json").write_text(
            json.dumps(
                {
                    "source_lm": source_lm.name,
                    "source_lm_sha256": _sha256(source_lm),
                    "quantized_lm": quantized_name,
                    "quantized_lm_sha256": _sha256(quantized_path),
                    "mlx_version": mx.__version__,
                    "quantization": spec.as_config(),
                },
                indent=2,
            )
            + "\n"
        )
        (destination / "README.md").write_text(_model_card(spec, config))
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def validate_quantization_request(source: Path, destination: Path) -> None:
    """Check that a conversion can create a new complete bundle."""
    config = _read_config(source.resolve())
    _source_files(source.resolve(), config)
    _ensure_new_destination(destination.resolve())


def validate_quantized_bundle(directory: Path) -> QuantizationSpec:
    """Ensure a directory is a complete, explicitly quantized bundle."""
    config = _read_config(directory)
    if QUANTIZATION_CONFIG_KEY not in config:
        raise ValueError(f"{directory} is not a quantized Hibiki bundle")
    spec = QuantizationSpec.from_config(config)
    assert spec is not None
    _source_files(directory, config)
    return spec


def _read_config(directory: Path) -> dict[str, object]:
    path = directory / "config.json"
    try:
        data = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    return data


def _source_files(directory: Path, config: dict[str, object]) -> tuple[Path, Path, Path]:
    names = []
    for key in ("moshi_name", "mimi_name", "tokenizer_name"):
        name = config.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f"{directory / 'config.json'} does not name its {key}")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"{path} is missing")
        names.append(path)
    return tuple(names)  # type: ignore[return-value]


def _ensure_new_destination(destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_card(spec: QuantizationSpec, config: dict[str, object]) -> str:
    return (
        "---\n"
        "license: cc-by-4.0\n"
        "language:\n"
        "- fr\n"
        "- en\n"
        "base_model: kyutai/hibiki-1b-mlx-bf16\n"
        "base_model_relation: quantized\n"
        "tags:\n"
        "- hibiki\n"
        "- mlx\n"
        "- speech-to-speech\n"
        "- quantized\n"
        "---\n\n"
        f"# Hibiki 1B MLX Q{spec.bits}\n\n"
        "Weight-only quantization of the language model from the artifact "
        f"revision `{config.get('model_revision', 'unknown')}`.\n\n"
        f"- Quantization: Q{spec.bits}, group size {spec.group_size}\n"
        "- Quantized modules: compatible MLX Linear layers only\n"
        "- Unchanged: Mimi codec, embeddings, normalisation layers, tokenizer\n"
    )
