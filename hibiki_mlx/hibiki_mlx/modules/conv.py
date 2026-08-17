"""Streaming convolutions used by the Mimi codec.

Adapted from `moshi_mlx.modules.conv` (Copyright (c) Kyutai, MIT licence). See
../../NOTICE.

MLX convolutions are NLC while the released weights and the codec's own tensor
layout are NCL, so every layer here transposes on the way in and out.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class Conv1d(nn.Module):
    """A 1D convolution over NCL input, with MLX's ``(out, ksize, in)`` weight."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        scale = 1 / (in_channels * ksize)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, ksize, in_channels // groups),
        )
        self.bias = mx.zeros(out_channels) if bias else None
        self._padding = padding
        self._groups = groups
        self._stride = stride
        self._dilation = dilation

    def __call__(self, xs: mx.array) -> mx.array:
        ys = mx.conv1d(
            xs.swapaxes(-1, -2),
            self.weight,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            groups=self._groups,
        )
        if self.bias is not None:
            ys = ys + self.bias
        return ys.swapaxes(-1, -2)


class ConvTranspose1d(nn.Module):
    """A transposed 1D convolution over NCL input.

    A depthwise transposed convolution is expanded into a dense one because MLX
    does not support grouped transposed convolutions. The expansion depends on
    the loaded weight, so ``update_in_place`` must run again after loading.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        scale = 1 / (in_channels * ksize)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels // groups, ksize, in_channels),
        )
        self.bias = mx.zeros(out_channels) if bias else None
        self._padding = padding
        self._groups = groups
        self._stride = stride
        self._ksize = ksize
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.update_in_place()

    def update_in_place(self) -> None:
        groups = self._groups
        if groups == self._in_channels and groups == self._out_channels:
            eye = mx.eye(self._out_channels).astype(self.weight.dtype)
            eye = eye.reshape((self._out_channels, 1, self._out_channels))
            eye = mx.repeat(eye, repeats=self._ksize, axis=1)
            self._expanded_weight = mx.repeat(self.weight, repeats=groups, axis=0) * eye
            self._expanded_groups = 1
        elif groups > 1:
            raise ValueError("only depthwise or dense transposed convolutions are supported")
        else:
            self._expanded_weight = self.weight
            self._expanded_groups = groups

    def __call__(self, xs: mx.array) -> mx.array:
        ys = mx.conv_transpose1d(
            xs.swapaxes(-1, -2),
            self._expanded_weight,
            stride=self._stride,
            padding=self._padding,
            groups=self._expanded_groups,
        )
        if self.bias is not None:
            ys = ys + self.bias
        return ys.swapaxes(-1, -2)


class NormConv1d(nn.Module):
    """The released weights keep an unnormalised convolution under ``conv``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.conv = Conv1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, xs: mx.array) -> mx.array:
        return self.conv(xs)


class NormConvTranspose1d(nn.Module):
    """The transposed counterpart of :class:`NormConv1d`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.convtr = ConvTranspose1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    @property
    def bias(self) -> mx.array | None:
        """The convolution's bias, so streaming callers need not reach for it."""
        return self.convtr.bias

    def __call__(self, xs: mx.array) -> mx.array:
        return self.convtr(xs)


def get_extra_padding_for_conv1d(
    xs: mx.array,
    ksize: int,
    stride: int,
    padding_total: int,
) -> int:
    length = xs.shape[-1]
    frames = max(length + padding_total - ksize, 0) / stride + 1.0
    ideal_length = (int(math.ceil(frames)) - 1) * stride + ksize - padding_total
    return max(0, ideal_length - length)


def unpad1d(xs: mx.array, unpad_l: int, unpad_r: int) -> mx.array:
    return xs[..., unpad_l : xs.shape[-1] - unpad_r]


class StreamableConv1d(nn.Module):
    """A causal convolution that carries its left context between steps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        dilation: int,
        groups: int,
        bias: bool,
        causal: bool,
        pad_mode: str,
    ):
        super().__init__()

        self.conv = NormConv1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self._causal = causal
        self._pad_mode = pad_mode
        self._ksize = ksize
        self._stride = stride
        self._dilation = dilation
        self._out_channels = out_channels
        self._prev_xs: mx.array | None = None
        self._left_pad_applied = False

    def reset_state(self) -> None:
        self._prev_xs = None
        self._left_pad_applied = False

    def __call__(self, xs: mx.array) -> mx.array:
        stride = self._stride
        ksize = (self._ksize - 1) * self._dilation + 1
        padding_total = ksize - stride
        extra_padding = get_extra_padding_for_conv1d(
            xs,
            ksize=ksize,
            stride=stride,
            padding_total=padding_total,
        )
        if self._causal:
            padding_left, padding_right = padding_total, 0
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
        none = (0, 0)
        widths = [none, none, (padding_left, padding_right + extra_padding)]
        return self.conv(mx.pad(xs, pad_width=widths, mode=self._pad_mode))

    def step(self, xs: mx.array) -> mx.array:
        batch, _, length = xs.shape
        if length == 0:
            return mx.zeros((batch, self._out_channels, 0))
        stride = self._stride
        ksize = (self._ksize - 1) * self._dilation + 1
        if not self._left_pad_applied:
            self._left_pad_applied = True
            xs = mx.pad(
                xs,
                pad_width=((0, 0), (0, 0), (ksize - stride, 0)),
                mode=self._pad_mode,
            )
        if self._prev_xs is not None:
            xs = mx.concatenate([self._prev_xs, xs], axis=-1)
        length = xs.shape[-1]
        frames = max(length + stride - ksize, 0) // stride
        if frames == 0:
            self._prev_xs = xs
            return mx.zeros((batch, self._out_channels, 0))
        self._prev_xs = xs[..., frames * stride :]
        return self.conv(xs[..., 0 : (frames - 1) * stride + ksize])


class StreamableConvTranspose1d(nn.Module):
    """A causal transposed convolution that carries its output tail."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int,
        bias: bool,
        causal: bool,
    ):
        super().__init__()

        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            ksize,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self._causal = causal
        self._ksize = ksize
        self._stride = stride
        self._out_channels = out_channels
        self._prev_ys: mx.array | None = None

    def reset_state(self) -> None:
        self._prev_ys = None

    def __call__(self, xs: mx.array) -> mx.array:
        padding_total = max(self._ksize - self._stride, 0)
        ys = self.convtr(xs)
        if self._causal:
            unpad_l, unpad_r = 0, padding_total
        else:
            unpad_r = padding_total // 2
            unpad_l = padding_total - unpad_r
        return unpad1d(ys, unpad_l=unpad_l, unpad_r=unpad_r)

    def step(self, xs: mx.array) -> mx.array:
        batch, _, length = xs.shape
        if length == 0:
            return mx.zeros((batch, self._out_channels, 0))
        ys = self.convtr(xs)
        produced = ys.shape[-1]
        if self._prev_ys is not None:
            prev_ys = self._prev_ys
            overlap = prev_ys.shape[-1]
            # The bias was already added to the overlapping tail, so remove it
            # before the two contributions are summed.
            if self.convtr.bias is not None:
                prev_ys = prev_ys - self.convtr.bias[None, :, None]
            ys = mx.concatenate([ys[..., :overlap] + prev_ys, ys[..., overlap:]], axis=-1)
        invalid_steps = self._ksize - self._stride
        ys, self._prev_ys = ys[..., : produced - invalid_steps], ys[..., produced - invalid_steps :]
        return ys


class ConvDownsample1d(nn.Module):
    """The 25 Hz to 12.5 Hz downsample in front of the quantizer."""

    def __init__(self, stride: int, dim: int, causal: bool):
        super().__init__()

        self.conv = StreamableConv1d(
            in_channels=dim,
            out_channels=dim,
            ksize=2 * stride,
            stride=stride,
            dilation=1,
            groups=1,
            bias=False,
            causal=causal,
            pad_mode="edge",
        )

    def reset_state(self) -> None:
        self.conv.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.conv(xs)

    def step(self, xs: mx.array) -> mx.array:
        return self.conv.step(xs)


class ConvTrUpsample1d(nn.Module):
    """The 12.5 Hz to 25 Hz upsample behind the quantizer."""

    def __init__(self, stride: int, dim: int, causal: bool):
        super().__init__()

        self.convtr = StreamableConvTranspose1d(
            in_channels=dim,
            out_channels=dim,
            ksize=2 * stride,
            stride=stride,
            groups=dim,
            bias=False,
            causal=causal,
        )

    def reset_state(self) -> None:
        self.convtr.reset_state()

    def __call__(self, xs: mx.array) -> mx.array:
        return self.convtr(xs)

    def step(self, xs: mx.array) -> mx.array:
        return self.convtr.step(xs)
