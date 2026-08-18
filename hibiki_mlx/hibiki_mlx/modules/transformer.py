"""The Transformer stack shared by the Hibiki model and the Mimi codec.

Only the variants the released bundle uses are implemented: self-attention with
optional RoPE, gated SiLU or plain GELU feed-forward, RMS or layer norm, and an
optional layer scale. Cross-attention is not part of this artifact bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .kv_cache import KVCache, RotatingKVCache


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    causal: bool
    layer_scale: float | None
    positional_embedding: str
    gating: bool
    norm: str
    context: int
    max_period: int
    max_seq_len: int
    conv_layout: bool
    bias_ff: bool = False
    bias_attn: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass
class LayerCache:
    """One layer's share of a session's attention state."""

    self_attn: KVCache | RotatingKVCache

    def reset(self) -> None:
        self.self_attn.reset()


class NoLayerScale(nn.Module):
    """Stands in for a layer scale the artifact bundle does not carry."""

    def __call__(self, xs: mx.array) -> mx.array:
        return xs


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.scale = mx.ones(dim)

    def __call__(self, xs: mx.array) -> mx.array:
        return xs * self.scale


def causal_mask(query_len: int, key_len: int, dtype: mx.Dtype) -> mx.array:
    """Forbid a query from attending to keys that follow it.

    Queries are the last ``query_len`` positions of the ``key_len`` keys, so
    query ``i`` may attend to keys up to ``key_len - query_len + i``.
    """
    queries = mx.arange(key_len - query_len, key_len)[:, None]
    keys = mx.arange(key_len)[None, :]
    return mx.where(keys <= queries, 0, -float("inf")).astype(dtype)


class Attention(nn.Module):
    """Self-attention over a bounded context, with fused QKV projection."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.in_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias_attn)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias_attn)
        self.cfg = cfg
        self.scale = cfg.head_dim ** (-0.5)
        self.rope = None
        if cfg.positional_embedding == "rope":
            self.rope = nn.RoPE(cfg.head_dim, traditional=True, base=cfg.max_period)
        elif cfg.positional_embedding != "none":
            raise ValueError(f"unsupported positional embedding {cfg.positional_embedding}")

    def __call__(self, xs: mx.array, cache: KVCache | RotatingKVCache) -> mx.array:
        batch, steps, dim = xs.shape
        qkv = self.in_proj(xs).reshape(batch, steps, 3, self.cfg.num_heads, self.cfg.head_dim)
        queries = qkv[:, :, 0].transpose(0, 2, 1, 3)
        keys = qkv[:, :, 1].transpose(0, 2, 1, 3)
        values = qkv[:, :, 2].transpose(0, 2, 1, 3)
        if self.rope is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)

        keys, values = cache.update_and_fetch(keys, values)
        key_len = keys.shape[2]
        target_len = steps + min(self.cfg.context, key_len - steps)
        if target_len < key_len:
            keys = keys[:, :, key_len - target_len :]
            values = values[:, :, key_len - target_len :]
            key_len = target_len

        # A single streaming step needs no mask: every cached key precedes it.
        mask = causal_mask(steps, key_len, xs.dtype) if self.cfg.causal and steps > 1 else None
        ys = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        ys = ys.transpose(0, 2, 1, 3).reshape(batch, steps, dim)
        return self.out_proj(ys)


class MlpGating(nn.Module):
    """A gated SiLU feed-forward whose two branches share one input projection."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        hidden = 2 * cfg.dim_feedforward // 3
        self.linear_in = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias_ff)
        self.linear_out = nn.Linear(hidden, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        xs = self.linear_in(xs)
        batch, steps, _ = xs.shape
        xs = xs.reshape(batch, steps, 2, -1)
        return self.linear_out(nn.silu(xs[:, :, 0]) * xs[:, :, 1])


class MlpNoGating(nn.Module):
    """The plain GELU feed-forward used by the Mimi Transformers."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward, bias=cfg.bias_ff)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model, bias=cfg.bias_ff)

    def __call__(self, xs: mx.array) -> mx.array:
        return self.linear2(nn.gelu_approx(self.linear1(xs)))


def _norm(cfg: TransformerConfig) -> nn.Module:
    if cfg.norm == "layer_norm":
        return nn.LayerNorm(cfg.d_model, 1e-5)
    if cfg.norm == "rms_norm":
        return nn.RMSNorm(cfg.d_model, 1e-8)
    raise ValueError(f"unsupported norm type {cfg.norm}")


class TransformerLayer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.gating = MlpGating(cfg) if cfg.gating else MlpNoGating(cfg)
        self.norm1 = _norm(cfg)
        self.norm2 = _norm(cfg)
        self.self_attn = Attention(cfg)
        if cfg.layer_scale is not None:
            self.layer_scale_1 = LayerScale(cfg.d_model)
            self.layer_scale_2 = LayerScale(cfg.d_model)
        else:
            self.layer_scale_1 = NoLayerScale()
            self.layer_scale_2 = NoLayerScale()

    def __call__(self, xs: mx.array, cache: LayerCache) -> mx.array:
        xs = xs + self.layer_scale_1(self.self_attn(self.norm1(xs), cache=cache.self_attn))
        return xs + self.layer_scale_2(self.gating(self.norm2(xs)))


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg
        self.layers = [TransformerLayer(cfg) for _ in range(cfg.num_layers)]

    def __call__(self, xs: mx.array, cache: list[LayerCache]) -> mx.array:
        for layer, layer_cache in zip(self.layers, cache):
            xs = layer(xs, cache=layer_cache)
        return xs

    def make_cache(self) -> list[LayerCache]:
        return [
            LayerCache(KVCache(head_dim=self.cfg.head_dim, n_kv_heads=self.cfg.num_heads))
            for _ in self.layers
        ]

    def make_rotating_cache(self) -> list[LayerCache]:
        """A cache bounded to ``max_seq_len`` positions, for the temporal stack."""
        return [
            LayerCache(
                RotatingKVCache(
                    head_dim=self.cfg.head_dim,
                    n_kv_heads=self.cfg.num_heads,
                    max_size=self.cfg.max_seq_len,
                )
            )
            for _ in self.layers
        ]


class ProjectedTransformer(nn.Module):
    """A Transformer with optional input and output projections.

    The Mimi Transformers run at the SEANet dimension, so both projections are
    absent from the released weights; ``conv_layout`` swaps the codec's NCL
    tensors into the Transformer's NLC layout.
    """

    def __init__(self, cfg: TransformerConfig, input_dim: int, output_dims: list[int]):
        super().__init__()

        self.conv_layout = cfg.conv_layout
        self.transformer = Transformer(cfg)
        self.input_proj = None
        if input_dim != cfg.d_model:
            self.input_proj = nn.Linear(input_dim, cfg.d_model, bias=False)
        self.output_projs = [
            None if output_dim == cfg.d_model else nn.Linear(cfg.d_model, output_dim, bias=False)
            for output_dim in output_dims
        ]

    def __call__(self, xs: mx.array, cache: list[LayerCache]) -> list[mx.array]:
        if self.conv_layout:
            xs = xs.swapaxes(1, 2)
        if self.input_proj is not None:
            xs = self.input_proj(xs)
        xs = self.transformer(xs, cache=cache)
        outs = []
        for output_proj in self.output_projs:
            out = xs if output_proj is None else output_proj(xs)
            outs.append(out.swapaxes(1, 2) if self.conv_layout else out)
        return outs

    def make_cache(self) -> list[LayerCache]:
        return self.transformer.make_cache()
