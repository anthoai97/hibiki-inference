"""The SEANet encoder and decoder that surround the Mimi Transformers.

Adapted from `moshi_mlx.modules.seanet` (Copyright (c) Kyutai, MIT licence).
See ../../NOTICE.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .conv import StreamableConv1d, StreamableConvTranspose1d


@dataclass
class SeanetConfig:
    dimension: int
    channels: int
    causal: bool
    nfilters: int
    nresidual_layers: int
    ratios: list[int]
    ksize: int
    residual_ksize: int
    last_ksize: int
    dilation_base: int
    pad_mode: str
    true_skip: bool
    compress: int


class StreamingAdd(nn.Module):
    """Adds two streams whose steps may produce different numbers of samples."""

    def __init__(self):
        super().__init__()

        self._lhs: mx.array | None = None
        self._rhs: mx.array | None = None

    def reset_state(self) -> None:
        self._lhs = None
        self._rhs = None

    def step(self, lhs: mx.array, rhs: mx.array) -> mx.array:
        if self._lhs is not None:
            lhs = mx.concatenate([self._lhs, lhs], axis=-1)
            self._lhs = None
        if self._rhs is not None:
            rhs = mx.concatenate([self._rhs, rhs], axis=-1)
            self._rhs = None
        lhs_length, rhs_length = lhs.shape[-1], rhs.shape[-1]
        if lhs_length == rhs_length:
            return lhs + rhs
        if lhs_length < rhs_length:
            self._rhs = rhs[..., lhs_length:]
            return lhs + rhs[..., :lhs_length]
        self._lhs = lhs[..., rhs_length:]
        return lhs[..., :rhs_length] + rhs


class SeanetResnetBlock(nn.Module):
    def __init__(self, cfg: SeanetConfig, dim: int, ksizes_and_dilations: list[tuple[int, int]]):
        super().__init__()

        hidden = dim // cfg.compress
        block = []
        for index, (ksize, dilation) in enumerate(ksizes_and_dilations):
            block.append(
                StreamableConv1d(
                    in_channels=dim if index == 0 else hidden,
                    out_channels=dim if index == len(ksizes_and_dilations) - 1 else hidden,
                    ksize=ksize,
                    stride=1,
                    dilation=dilation,
                    groups=1,
                    bias=True,
                    causal=cfg.causal,
                    pad_mode=cfg.pad_mode,
                )
            )
        self.block = block
        self.streaming_add = StreamingAdd()
        self.shortcut = None
        if not cfg.true_skip:
            self.shortcut = StreamableConv1d(
                in_channels=dim,
                out_channels=dim,
                ksize=1,
                stride=1,
                dilation=1,
                groups=1,
                bias=True,
                causal=cfg.causal,
                pad_mode=cfg.pad_mode,
            )

    def reset_state(self) -> None:
        if self.shortcut is not None:
            self.shortcut.reset_state()
        self.streaming_add.reset_state()
        for conv in self.block:
            conv.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        residual = xs
        for conv in self.block:
            xs = conv(nn.elu(xs, alpha=1.0))
        if self.shortcut is None:
            return xs + residual
        return xs + self.shortcut(residual)

    def step(self, xs: mx.array) -> mx.array:
        residual = xs
        for conv in self.block:
            xs = conv.step(nn.elu(xs, alpha=1.0))
        if self.shortcut is None:
            return self.streaming_add.step(xs, residual)
        return self.streaming_add.step(xs, self.shortcut.step(residual))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: SeanetConfig, ratio: int, mult: int):
        super().__init__()

        residuals = []
        dilation = 1
        for _ in range(cfg.nresidual_layers):
            residuals.append(
                SeanetResnetBlock(
                    cfg,
                    dim=mult * cfg.nfilters,
                    ksizes_and_dilations=[(cfg.residual_ksize, dilation), (1, 1)],
                )
            )
            dilation *= cfg.dilation_base
        self.residuals = residuals
        self.downsample = StreamableConv1d(
            in_channels=mult * cfg.nfilters,
            out_channels=mult * cfg.nfilters * 2,
            ksize=ratio * 2,
            stride=ratio,
            dilation=1,
            groups=1,
            bias=True,
            causal=True,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self) -> None:
        self.downsample.reset_state()
        for residual in self.residuals:
            residual.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        for residual in self.residuals:
            xs = residual(xs)
        return self.downsample(nn.elu(xs, alpha=1.0))

    def step(self, xs: mx.array) -> mx.array:
        for residual in self.residuals:
            xs = residual.step(xs)
        return self.downsample.step(nn.elu(xs, alpha=1.0))


class SeanetEncoder(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        super().__init__()

        mult = 1
        self.init_conv1d = StreamableConv1d(
            in_channels=cfg.channels,
            out_channels=mult * cfg.nfilters,
            ksize=cfg.ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )
        layers = []
        for ratio in reversed(cfg.ratios):
            layers.append(EncoderLayer(cfg, ratio=ratio, mult=mult))
            mult *= 2
        self.layers = layers
        self.final_conv1d = StreamableConv1d(
            in_channels=mult * cfg.nfilters,
            out_channels=cfg.dimension,
            ksize=cfg.last_ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self) -> None:
        self.init_conv1d.reset_state()
        self.final_conv1d.reset_state()
        for layer in self.layers:
            layer.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d(xs)
        for layer in self.layers:
            xs = layer(xs)
        return self.final_conv1d(nn.elu(xs, alpha=1.0))

    def step(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d.step(xs)
        for layer in self.layers:
            xs = layer.step(xs)
        return self.final_conv1d.step(nn.elu(xs, alpha=1.0))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: SeanetConfig, ratio: int, mult: int):
        super().__init__()

        self.upsample = StreamableConvTranspose1d(
            in_channels=mult * cfg.nfilters,
            out_channels=mult * cfg.nfilters // 2,
            ksize=ratio * 2,
            stride=ratio,
            groups=1,
            bias=True,
            causal=cfg.causal,
        )
        residuals = []
        dilation = 1
        for _ in range(cfg.nresidual_layers):
            residuals.append(
                SeanetResnetBlock(
                    cfg,
                    dim=mult * cfg.nfilters // 2,
                    ksizes_and_dilations=[(cfg.residual_ksize, dilation), (1, 1)],
                )
            )
            dilation *= cfg.dilation_base
        self.residuals = residuals

    def reset_state(self) -> None:
        self.upsample.reset_state()
        for residual in self.residuals:
            residual.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.upsample(nn.elu(xs, alpha=1.0))
        for residual in self.residuals:
            xs = residual(xs)
        return xs

    def step(self, xs: mx.array) -> mx.array:
        xs = self.upsample.step(nn.elu(xs, alpha=1.0))
        for residual in self.residuals:
            xs = residual.step(xs)
        return xs


class SeanetDecoder(nn.Module):
    def __init__(self, cfg: SeanetConfig):
        super().__init__()

        mult = 1 << len(cfg.ratios)
        self.init_conv1d = StreamableConv1d(
            in_channels=cfg.dimension,
            out_channels=mult * cfg.nfilters,
            ksize=cfg.ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )
        layers = []
        for ratio in cfg.ratios:
            layers.append(DecoderLayer(cfg, ratio=ratio, mult=mult))
            mult //= 2
        self.layers = layers
        self.final_conv1d = StreamableConv1d(
            in_channels=cfg.nfilters,
            out_channels=cfg.channels,
            ksize=cfg.last_ksize,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=cfg.causal,
            pad_mode=cfg.pad_mode,
        )

    def reset_state(self) -> None:
        self.init_conv1d.reset_state()
        self.final_conv1d.reset_state()
        for layer in self.layers:
            layer.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d(xs)
        for layer in self.layers:
            xs = layer(xs)
        return self.final_conv1d(nn.elu(xs, alpha=1.0))

    def step(self, xs: mx.array) -> mx.array:
        xs = self.init_conv1d.step(xs)
        for layer in self.layers:
            xs = layer.step(xs)
        return self.final_conv1d.step(nn.elu(xs, alpha=1.0))
