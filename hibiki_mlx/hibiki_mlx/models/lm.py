"""The Hibiki language model: Temporal Transformer plus Depth Transformer.

Adapted from `moshi_mlx.models.lm` (Copyright (c) Kyutai, MIT licence). See
../../NOTICE.

The released ``hibiki-mlx-*.safetensors`` is already in MLX naming, so the
parameter tree built here is the load contract: every name and shape must match
the file exactly.

Unlike the reference, the model does not own its attention caches. Caches are
session state, so ``make_transformer_cache()`` and ``make_depformer_cache()``
hand them out instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..modules.conditioner import ConditionProvider, ConditionTensor, LutConditionerConfig
from ..modules.transformer import LayerCache, Transformer, TransformerConfig
from ..sampling import Sampler

# The config names its norm ``rms_norm_f32``; the suffix describes the
# accumulation precision, not a different parameter layout.
_SUPPORTED_NORMS = ("rms_norm", "rms_norm_f32")


@dataclass
class DepFormerConfig:
    transformer: TransformerConfig
    num_slices: int


@dataclass
class LmConfig:
    transformer: TransformerConfig
    depformer: DepFormerConfig
    text_in_vocab_size: int
    text_out_vocab_size: int
    text_padding_token: int
    audio_vocab_size: int
    audio_codebooks: int
    audio_delays: list[int]
    conditioners: dict[str, LutConditionerConfig]

    @property
    def target_codebooks(self) -> int:
        """The target-audio codebooks the Depth Transformer samples."""
        return self.depformer.num_slices

    @property
    def source_codebooks(self) -> int:
        """The source-audio codebooks Mimi supplies from the French input."""
        return self.audio_codebooks - self.target_codebooks

    @property
    def audio_padding_token(self) -> int:
        return self.audio_vocab_size - 1

    @classmethod
    def from_config_dict(cls, data: dict) -> "LmConfig":
        """Build the model contract from the bundle's own ``config.json``."""
        if data["norm"] not in _SUPPORTED_NORMS:
            raise ValueError(f"unsupported norm {data['norm']!r}")
        if data["gating"] != "silu":
            raise ValueError(f"unsupported gating {data['gating']!r}")
        if not data.get("depformer_weights_per_step", False):
            raise ValueError("this implementation expects per-step Depth Transformer weights")
        if len(data["delays"]) != data["n_q"] + 1:
            raise ValueError(
                f"expected {data['n_q'] + 1} delays for one text and "
                f"{data['n_q']} audio streams, found {len(data['delays'])}"
            )

        transformer = TransformerConfig(
            d_model=data["dim"],
            num_heads=data["num_heads"],
            num_layers=data["num_layers"],
            dim_feedforward=int(data["hidden_scale"] * data["dim"]),
            causal=data["causal"],
            layer_scale=data["layer_scale"],
            context=data["context"],
            max_period=data["max_period"],
            max_seq_len=4096,
            gating=True,
            norm="rms_norm",
            positional_embedding=data["positional_embedding"],
            conv_layout=False,
        )
        depformer = DepFormerConfig(
            transformer=TransformerConfig(
                d_model=data["depformer_dim"],
                num_heads=data["depformer_num_heads"],
                num_layers=data["depformer_num_layers"],
                dim_feedforward=data["depformer_dim_feedforward"],
                causal=data["depformer_causal"],
                layer_scale=data["depformer_layer_scale"],
                context=data["depformer_context"],
                max_period=data["depformer_max_period"],
                max_seq_len=4096,
                gating=True,
                norm="rms_norm",
                positional_embedding=data["depformer_pos_emb"],
                conv_layout=False,
            ),
            num_slices=data["dep_q"],
        )
        conditioners = {}
        for name, conditioner in data.get("conditioners", {}).items():
            if conditioner["type"] != "lut":
                raise ValueError(f"unsupported conditioner type {conditioner['type']!r}")
            lut = conditioner["lut"]
            conditioners[name] = LutConditionerConfig(
                n_bins=lut["n_bins"],
                dim=lut["dim"],
                tokenizer=lut["tokenizer"],
                possible_values=lut["possible_values"],
            )
        return cls(
            transformer=transformer,
            depformer=depformer,
            text_in_vocab_size=data["text_card"] + 1,
            text_out_vocab_size=data["text_card"],
            # The bundle names its own no-text id; do not assume the usual 3.
            text_padding_token=data["existing_text_padding_id"],
            audio_vocab_size=data["card"] + 1,
            audio_codebooks=data["n_q"],
            # The first delay belongs to the text stream.
            audio_delays=data["delays"][1:],
            conditioners=conditioners,
        )


class ScaledEmbedding(nn.Embedding):
    """An embedding that maps ``zero_idx`` to an all-zero row.

    The reference scheduler feeds ``-1`` for "no input at this position", which
    must contribute nothing to the summed input embedding. This implementation's
    schedule never does: it feeds the audio padding token instead, and refuses
    to read a position nothing has written. The behaviour is kept so the module
    stays interchangeable with the reference, not because anything relies on it.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, zero_idx: int = -1):
        super().__init__(num_embeddings, embedding_dim)

        if zero_idx >= 0:
            raise ValueError("zero_idx must be negative so it cannot be a real token")
        self.zero_idx = zero_idx

    def __call__(self, input: mx.array) -> mx.array:
        is_zero = input == self.zero_idx
        ys = self.weight[mx.maximum(input, 0)]
        return mx.where(is_zero[..., None], mx.zeros(1, dtype=ys.dtype), ys)


class DepFormerSlice(nn.Module):
    """One depth position: its own embedding, projections, and Transformer."""

    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        main_transformer_dim: int,
        cfg: DepFormerConfig,
    ):
        super().__init__()

        dim = cfg.transformer.d_model
        self.emb = ScaledEmbedding(in_vocab_size, dim)
        self.linear_in = nn.Linear(main_transformer_dim, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_vocab_size, bias=False)
        self.transformer = Transformer(cfg.transformer)


class DepFormer(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        self.slices = [
            DepFormerSlice(
                # The first slice is conditioned on the sampled text token, the
                # rest on the previous codebook's audio token.
                cfg.text_in_vocab_size if index == 0 else cfg.audio_vocab_size,
                cfg.audio_vocab_size - 1,
                main_transformer_dim=cfg.transformer.d_model,
                cfg=cfg.depformer,
            )
            for index in range(cfg.depformer.num_slices)
        ]

    def make_cache(self) -> list[LayerCache]:
        """One scratch cache, shared by the slices and reset every frame."""
        return self.slices[0].transformer.make_cache()

    def sample(
        self,
        transformer_out: mx.array,
        text_token: mx.array,
        cache: list[LayerCache],
        sampler: Sampler,
    ) -> mx.array:
        """Sample the eight target codebooks for one frame, in depth order.

        Each slice is conditioned on the temporal state and on the token the
        previous slice produced, so the cache is scratch state for this frame
        alone and is cleared before the walk starts.
        """
        for layer_cache in cache:
            layer_cache.reset()
        tokens = []
        last_token = text_token
        for slice_ in self.slices:
            xs = slice_.linear_in(transformer_out) + slice_.emb(last_token)
            xs = slice_.transformer(xs, cache=cache)
            last_token = sampler(slice_.linear_out(xs))
            tokens.append(last_token)
        return mx.stack(tokens, axis=1)


class Lm(nn.Module):
    def __init__(self, cfg: LmConfig):
        super().__init__()

        dim = cfg.transformer.d_model
        self.cfg = cfg
        self.transformer = Transformer(cfg.transformer)
        self.depformer = DepFormer(cfg)
        self.text_emb = ScaledEmbedding(cfg.text_in_vocab_size, dim)
        self.out_norm = nn.RMSNorm(dim, 1e-8)
        self.text_linear = nn.Linear(dim, cfg.text_out_vocab_size, bias=False)
        self.audio_embs = [
            ScaledEmbedding(cfg.audio_vocab_size, dim) for _ in range(cfg.audio_codebooks)
        ]
        self.condition_provider = None
        if cfg.conditioners:
            self.condition_provider = ConditionProvider(dim, cfg.conditioners)

    def make_transformer_cache(self) -> list[LayerCache]:
        return self.transformer.make_rotating_cache()

    def make_depformer_cache(self) -> list[LayerCache]:
        return self.depformer.make_cache()

    def forward_text(
        self,
        token_ids: mx.array,
        cache: list[LayerCache],
    ) -> tuple[mx.array, mx.array]:
        """Run one temporal step, returning its state and the text logits."""
        transformer_out = self.out_norm(self.transformer(self.text_emb(token_ids), cache=cache))
        return transformer_out, self.text_linear(transformer_out)

    def sample_step(
        self,
        text_token_ids: mx.array,
        audio_token_ids: list[mx.array],
        transformer_cache: list[LayerCache],
        depformer_cache: list[LayerCache],
        text_sampler: Sampler,
        audio_sampler: Sampler,
        condition: ConditionTensor | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Advance one frame: sample the text token, then the audio codebooks.

        ``text_token_ids`` is ``[B, 1]`` and ``audio_token_ids`` holds one
        ``[B, 1]`` column per audio stream, already placed at its delayed
        position by the caller. The returned audio tokens are ``[B, dep_q, 1]``.
        """
        xs = self.text_emb(text_token_ids)
        for token_ids, embedding in zip(audio_token_ids, self.audio_embs):
            xs = xs + embedding(token_ids)
        if condition is not None:
            xs = xs + mx.expand_dims(condition.tensor, axis=1)

        transformer_out = self.out_norm(self.transformer(xs, cache=transformer_cache))
        text_token = text_sampler(self.text_linear(transformer_out))
        audio_tokens = self.depformer.sample(
            transformer_out,
            text_token,
            depformer_cache,
            audio_sampler,
        )
        return text_token, audio_tokens
