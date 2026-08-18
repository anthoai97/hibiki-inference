"""The compatible weight-only Q8/Q4 contract for a Hibiki artifact bundle."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.nn as nn

QUANTIZATION_CONFIG_KEY = "hibiki_mlx_quantization"
QUANTIZATION_FORMAT = "linear-v1"


@dataclass(frozen=True)
class QuantizationSpec:
    """The quantized linear layout required to load one artifact bundle."""

    bits: int
    group_size: int

    @classmethod
    def for_bits(cls, bits: int) -> "QuantizationSpec":
        group_sizes = {8: 64, 4: 32}
        try:
            return cls(bits=bits, group_size=group_sizes[bits])
        except KeyError as error:
            raise ValueError("only Q8 and Q4 artifacts are supported") from error

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "QuantizationSpec | None":
        raw = config.get(QUANTIZATION_CONFIG_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{QUANTIZATION_CONFIG_KEY} must be an object")
        if raw.get("format") != QUANTIZATION_FORMAT:
            raise ValueError(f"{QUANTIZATION_CONFIG_KEY} has an unsupported format")
        bits = raw.get("bits")
        group_size = raw.get("group_size")
        if not isinstance(bits, int) or not isinstance(group_size, int):
            raise ValueError(f"{QUANTIZATION_CONFIG_KEY} must declare integer bits and group_size")
        spec = cls.for_bits(bits)
        if group_size != spec.group_size:
            raise ValueError(f"Q{bits} requires group_size={spec.group_size}, not {group_size}")
        return spec

    def as_config(self) -> dict[str, object]:
        return {
            "format": QUANTIZATION_FORMAT,
            "target": "lm-linear",
            "bits": self.bits,
            "group_size": self.group_size,
        }


def quantize_linear_layers(model: nn.Module, spec: QuantizationSpec) -> None:
    """Replace only Linear leaves that can use the packed MLX layout.

    The model's ``ScaledEmbedding`` leaves deliberately remain unquantized:
    their vocabulary shapes are incompatible with MLX quantization and their
    padding behaviour is part of the streaming contract.
    """

    def compatible_linear(_path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        weight = module.weight
        return (
            weight.ndim == 2
            and all(dimension % 32 == 0 for dimension in weight.shape)
            and weight.shape[-1] % spec.group_size == 0
        )

    nn.quantize(
        model,
        bits=spec.bits,
        group_size=spec.group_size,
        class_predicate=compatible_linear,
    )
