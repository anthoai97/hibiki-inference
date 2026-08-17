"""Key/value caches for the streaming attention layers.

Adapted from `moshi_mlx.modules.kv_cache`, which in turn comes from mlx-examples
(Copyright (c) 2023-2024 Apple Inc.). See ../../NOTICE.

The caches are session state, not model parameters: nothing here is loaded from
the artifact bundle.
"""

from __future__ import annotations

import mlx.core as mx


class KVCache:
    """A cache that grows in fixed steps and keeps every past position."""

    def __init__(self, head_dim: int, n_kv_heads: int, step: int = 256):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        previous = self.offset
        if self.keys is None or (previous + keys.shape[2]) > self.keys.shape[2]:
            batch = keys.shape[0]
            steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (batch, self.n_kv_heads, steps * self.step, self.head_dim)
            new_keys = mx.zeros(shape, keys.dtype)
            new_values = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                if previous % self.step != 0:
                    self.keys = self.keys[..., :previous, :]
                    self.values = self.values[..., :previous, :]
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)
            else:
                self.keys, self.values = new_keys, new_values

        assert self.values is not None
        self.offset += keys.shape[2]
        self.keys[..., previous : self.offset, :] = keys
        self.values[..., previous : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def reset(self) -> None:
        self.offset = 0
        self.keys = None
        self.values = None


class RotatingKVCache:
    """A cache bounded to ``max_size`` positions that overwrites the oldest one.

    ``offset`` keeps counting past ``max_size`` so absolute RoPE positions stay
    correct after the buffer wraps.
    """

    def __init__(
        self,
        head_dim: int,
        n_kv_heads: int,
        max_size: int,
        keep: int = 0,
        step: int = 256,
    ):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keep = keep
        self.max_size = max_size
        self.step = step
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self._idx = 0

    def _trim(self, trim_size: int, values: mx.array, append: mx.array | None = None) -> mx.array:
        if trim_size > 0:
            to_concatenate = [values[..., : self.keep, :], values[..., trim_size + self.keep :, :]]
        else:
            to_concatenate = [values]
        if append is not None:
            to_concatenate.append(append)
        return mx.concatenate(to_concatenate, axis=2)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        previous = self.offset
        batch, _, steps = keys.shape[:3]

        if steps > 1:
            if self.keys is None:
                self.keys = keys
                self.values = values
            else:
                # Leave max_size - 1 positions so every new token still sees a
                # full context window.
                trim_size = self.keys.shape[2] - self.max_size + 1
                self.keys = self._trim(trim_size, self.keys, keys)
                self.values = self._trim(trim_size, self.values, values)
            self.offset += steps
            self._idx = self.keys.shape[2]
            return self.keys, self.values

        if self.keys is None or (
            previous >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - previous)
            shape = (batch, self.n_kv_heads, new_size, self.head_dim)
            new_keys = mx.zeros(shape, keys.dtype)
            new_values = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                assert self.values is not None
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)
            else:
                self.keys, self.values = new_keys, new_values
            self._idx = previous

        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = self.keep

        assert self.values is not None
        self.keys[..., self._idx : self._idx + 1, :] = keys
        self.values[..., self._idx : self._idx + 1, :] = values
        self.offset += 1
        self._idx += 1

        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def reset(self) -> None:
        self.offset = 0
        self._idx = 0
        self.keys = None
        self.values = None
