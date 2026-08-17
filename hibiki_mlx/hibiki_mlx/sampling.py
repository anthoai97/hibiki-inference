"""Token sampling for the text and audio heads.

Adapted from `moshi_mlx.utils.sampling`, which comes from mlx-examples
(Copyright (c) 2023-2024 Apple Inc.). See ../NOTICE.

Sampling draws on MLX's global random state, so a translation is reproducible
only for a fixed seed and a fixed order of calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import mlx.core as mx


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(logits: mx.array, top_k: int, temp: float) -> mx.array:
    """Sample from the ``top_k`` most likely tokens."""
    logits = logits * (1 / temp)
    masked = mx.argpartition(-logits, kth=top_k - 1, axis=-1)[..., top_k:]
    logits = mx.put_along_axis(logits, masked, mx.array(-float("inf"), logits.dtype), axis=-1)
    return mx.random.categorical(logits, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_p_sampling(logits: mx.array, top_p: float, temp: float) -> mx.array:
    """Sample from the smallest set of tokens whose mass reaches ``top_p``."""
    probs = mx.softmax(logits * (1 / temp), axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)
    kept = mx.where(cumulative > 1 - top_p, sorted_probs, 0)
    sampled = mx.random.categorical(mx.log(kept), axis=-1)
    return mx.take_along_axis(sorted_indices, sampled[..., None], axis=-1).squeeze(-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    return mx.random.categorical(logits * (1 / temp), axis=-1)


@dataclass(frozen=True)
class Sampler:
    """Turns one head's logits into one token per batch entry."""

    temp: float = 0.8
    top_k: int | None = None
    top_p: float = 0.0

    def __call__(self, logits: mx.array) -> mx.array:
        if self.temp == 0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        if self.top_k is not None and self.top_k > 0:
            token = top_k_sampling(logits, self.top_k, self.temp)
        elif 0 < self.top_p < 1.0:
            token = top_p_sampling(logits, self.top_p, self.temp)
        else:
            token = categorical_sampling(logits, self.temp)
        return token.astype(mx.int32)
