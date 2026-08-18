"""The Mimi codec: SEANet, streaming Transformers, and a split RVQ.

The bundle's ``config.json`` says nothing about the codec, so the architecture
below is this implementation's explicit contract for
``mimi-dbaa9758@125.safetensors``. Unlike the Hibiki weights, that file is in
PyTorch naming and layout, so :func:`remap_released_weights` converts it before
the strict load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.conv import ConvDownsample1d, ConvTranspose1d, ConvTrUpsample1d
from ..modules.quantization import EuclideanCodebook, SplitResidualVectorQuantizer
from ..modules.seanet import SeanetConfig, SeanetDecoder, SeanetEncoder
from ..modules.transformer import LayerCache, ProjectedTransformer, TransformerConfig


@dataclass
class MimiConfig:
    channels: int
    sample_rate: float
    frame_rate: float
    seanet: SeanetConfig
    transformer: TransformerConfig
    quantizer_nq: int
    quantizer_bins: int
    quantizer_dim: int

    @property
    def frame_size(self) -> int:
        """The 1,920 samples of 24 kHz PCM that make up one 80 ms frame."""
        return int(self.sample_rate / self.frame_rate)


def mimi_202407(num_codebooks: int) -> MimiConfig:
    """The released causal 24 kHz Mimi at 12.5 Hz."""
    seanet = SeanetConfig(
        dimension=512,
        channels=1,
        causal=True,
        nfilters=64,
        nresidual_layers=1,
        ratios=[8, 6, 5, 4],
        ksize=7,
        residual_ksize=3,
        last_ksize=3,
        dilation_base=2,
        pad_mode="constant",
        true_skip=True,
        compress=2,
    )
    transformer = TransformerConfig(
        d_model=seanet.dimension,
        num_heads=8,
        num_layers=8,
        dim_feedforward=2048,
        causal=True,
        layer_scale=0.01,
        context=250,
        max_period=10000,
        max_seq_len=8192,
        gating=False,
        norm="layer_norm",
        positional_embedding="rope",
        conv_layout=True,
    )
    return MimiConfig(
        channels=1,
        sample_rate=24000,
        frame_rate=12.5,
        seanet=seanet,
        transformer=transformer,
        quantizer_nq=num_codebooks,
        quantizer_bins=2048,
        quantizer_dim=256,
    )


class Mimi(nn.Module):
    """The codec's weights, plus the convolution state one session streams through.

    Attention caches are handed out by ``make_encoder_cache`` and
    ``make_decoder_cache`` rather than owned here. The streaming convolution
    state still lives inside the layers, so one loaded codec drives one session
    until that state is lifted out too.
    """

    def __init__(self, cfg: MimiConfig):
        super().__init__()

        dim = cfg.seanet.dimension
        self.cfg = cfg
        encoder_frame_rate = cfg.sample_rate / math.prod(cfg.seanet.ratios)
        downsample_stride = int(encoder_frame_rate / cfg.frame_rate)
        self.encoder = SeanetEncoder(cfg.seanet)
        self.decoder = SeanetDecoder(cfg.seanet)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=cfg.quantizer_dim,
            input_dim=dim,
            output_dim=dim,
            nq=cfg.quantizer_nq,
            bins=cfg.quantizer_bins,
        )
        self.encoder_transformer = ProjectedTransformer(
            cfg.transformer, input_dim=dim, output_dims=[dim]
        )
        self.decoder_transformer = ProjectedTransformer(
            cfg.transformer, input_dim=dim, output_dims=[dim]
        )
        self.downsample = ConvDownsample1d(stride=downsample_stride, dim=dim, causal=True)
        self.upsample = ConvTrUpsample1d(stride=downsample_stride, dim=dim, causal=True)

    def make_encoder_cache(self) -> list[LayerCache]:
        return self.encoder_transformer.make_cache()

    def make_decoder_cache(self) -> list[LayerCache]:
        return self.decoder_transformer.make_cache()

    def reset_state(self) -> None:
        """Clear the streaming convolution state held inside the layers."""
        self.encoder.reset_state()
        self.decoder.reset_state()
        self.downsample.reset_state()
        self.upsample.reset_state()

    def encode_step(self, xs: mx.array, cache: list[LayerCache]) -> mx.array:
        """Turn one PCM step ``[B, 1, T]`` into audio codes ``[B, n_q, T']``."""
        xs = self.encoder.step(xs)
        xs = self.encoder_transformer(xs, cache=cache)[0]
        return self.quantizer.encode(self.downsample.step(xs))

    def decode_step(self, xs: mx.array, cache: list[LayerCache]) -> mx.array:
        """Turn one code step ``[B, n_q, T']`` back into PCM ``[B, 1, T]``."""
        xs = self.upsample.step(self.quantizer.decode(xs))
        xs = self.decoder_transformer(xs, cache=cache)[0]
        return self.decoder.step(xs)

    def refresh_derived_state(self) -> None:
        """Recompute state that is derived from the loaded weights.

        MLX's ``update`` writes parameters straight into the tree without
        calling any layer's own hook, so the codebook centroids and the
        expanded depthwise transposed convolutions have to be rebuilt here
        after every load.
        """

        def refresh(_name: str, module: nn.Module) -> None:
            if isinstance(module, (EuclideanCodebook, ConvTranspose1d)):
                module.update_in_place()

        self.apply_to_modules(refresh)


_SEANET_BLOCK_INDEX = {"1": "0", "3": "1"}


def _seanet_layout(upsampling: bool) -> dict[int, str]:
    """Map a released SEANet sequence index onto this implementation's name.

    Both stacks are one initial convolution, four resampling stages, and one
    final convolution; the released files number them by their position in a
    PyTorch ``Sequential`` that also holds the activations.
    """
    layout = {0: "init_conv1d", 14: "final_conv1d"}
    for index in range(4):
        if upsampling:
            layout[2 + 3 * index] = f"layers.{index}.upsample"
            layout[3 + 3 * index] = f"layers.{index}.residuals.0"
        else:
            layout[1 + 3 * index] = f"layers.{index}.residuals.0"
            layout[3 + 3 * index] = f"layers.{index}.downsample"
    return layout


_ENCODER_LAYOUT = _seanet_layout(upsampling=False)
_DECODER_LAYOUT = _seanet_layout(upsampling=True)


class MimiWeightError(ValueError):
    """Raised when a released parameter has no place in this implementation."""


def _remap_name(name: str) -> str:
    # The released names mark private submodules with a leading underscore,
    # which MLX would read as "not a parameter".
    name = ".".join(part.removeprefix("_") for part in name.split("."))

    for side, layout in (("encoder", _ENCODER_LAYOUT), ("decoder", _DECODER_LAYOUT)):
        prefix = f"{side}.model."
        if not name.startswith(prefix):
            continue
        index, _, rest = name[len(prefix) :].partition(".")
        target = layout.get(int(index))
        if target is None:
            raise MimiWeightError(f"{name} is at an unexpected {side} position {index}")
        parts = rest.split(".")
        if parts[0] == "block":
            # The residual block's activations are dropped, so its two
            # convolutions move down to indices 0 and 1.
            block_index = _SEANET_BLOCK_INDEX.get(parts[1])
            if block_index is None:
                raise MimiWeightError(f"{name} is at an unexpected residual position {parts[1]}")
            parts[1] = block_index
        name = f"{side}.{target}." + ".".join(parts)
        break

    if name.endswith(".in_proj_weight"):
        name = name.removesuffix(".in_proj_weight") + ".in_proj.weight"
    for linear in ("linear1", "linear2"):
        if name.endswith(f".{linear}.weight"):
            name = name.removesuffix(f".{linear}.weight") + f".gating.{linear}.weight"
    return name


def _remap_value(name: str, value: mx.array) -> mx.array:
    # PyTorch convolutions are (out, in, ksize) and MLX's are (out, ksize, in);
    # transposed convolutions are (in, out, ksize) against MLX's (out, ksize, in).
    if name.endswith(".conv.weight") or (
        name.startswith("quantizer.") and name.endswith(("input_proj.weight", "output_proj.weight"))
    ):
        return value.swapaxes(-1, -2)
    if name.endswith(".convtr.weight"):
        return value.transpose(1, 2, 0)
    return value


def remap_released_weights(
    raw: dict[str, mx.array],
    *,
    codebooks: int,
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    """Convert released codec weights into this implementation's parameters.

    The released file carries every codebook Mimi was trained with. Hibiki uses
    only the first ``codebooks`` of them, so the rest are dropped and returned
    so a caller can report exactly what was left behind. Anything else that
    does not belong is an error, not a silent drop.
    """
    kept_rest = codebooks - 1
    rest_prefix = "quantizer.rvq_rest.vq.layers."
    weights: dict[str, mx.array] = {}
    dropped: list[str] = []
    for name, value in raw.items():
        mapped = _remap_name(name)
        if mapped.startswith(rest_prefix):
            index = int(mapped[len(rest_prefix) :].split(".", 1)[0])
            if index >= kept_rest:
                dropped.append(name)
                continue
        weights[mapped] = _remap_value(mapped, value)
    return weights, tuple(sorted(dropped))
