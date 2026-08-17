"""The lookup-table conditioner that carries Hibiki's quality label.

The released bundle declares one ``description`` conditioner whose value is a
label such as ``very_good``. Its projected embedding is added at every time
step, so the same tensor is reused for a whole session.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LutConditionerConfig:
    n_bins: int
    dim: int
    tokenizer: str
    possible_values: list[str]


@dataclass
class ConditionTensor:
    tensor: mx.array


class LutConditioner(nn.Module):
    def __init__(self, output_dim: int, cfg: LutConditionerConfig):
        super().__init__()

        if cfg.tokenizer != "noop":
            raise ValueError(f"unsupported conditioner tokenizer {cfg.tokenizer}")
        self.embed = nn.Embedding(cfg.n_bins + 1, cfg.dim)
        self.output_proj = nn.Linear(cfg.dim, output_dim, bias=False)
        self.learnt_padding = mx.zeros((1, 1, output_dim))
        self.possible_values = {value: index for index, value in enumerate(cfg.possible_values)}

    def condition(self, value: str) -> mx.array:
        index = self.possible_values.get(value)
        if index is None:
            raise ValueError(
                f"unknown condition {value!r}; expected one of {sorted(self.possible_values)}"
            )
        return self.output_proj(self.embed(mx.array([index])))


class ConditionProvider(nn.Module):
    def __init__(self, output_dim: int, cfg: dict[str, LutConditionerConfig]):
        super().__init__()

        self.conditioners = {
            name: LutConditioner(output_dim, conditioner) for name, conditioner in cfg.items()
        }

    def condition_tensor(self, name: str, value: str) -> ConditionTensor:
        if name not in self.conditioners:
            raise ValueError(f"unknown conditioner {name!r}")
        return ConditionTensor(self.conditioners[name].condition(value))
