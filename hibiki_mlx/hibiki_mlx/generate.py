"""The delayed-stream schedule that drives one generation step at a time.

Seven of the eight codebooks in each audio stream are delayed by two frames, so
the tokens fed to the model at step ``t`` come from three different positions on
the model timeline, and the audio that becomes complete at step ``t`` belongs to
frame ``t - 2``.

Where the reference allocates the whole schedule up front and therefore needs a
step limit, this keeps only the positions still in play, so a run's length is
not bounded by its scheduler.
"""

from __future__ import annotations

import mlx.core as mx

from .models.lm import Lm
from .modules.conditioner import ConditionTensor
from .modules.transformer import LayerCache
from .sampling import Sampler

# Marks a position that should have been written before it was read. It can
# never collide with a real token, which is why it is negative.
UNGENERATED_TOKEN = -2


class ScheduleError(RuntimeError):
    """Raised when the schedule is asked for a token nothing has produced."""


class LmGen:
    """The delayed token schedule for one run, with its caches and samplers.

    Classifier-free guidance is deliberately absent. Until MLX is shown to hold
    distinct ``very_good`` and ``very_bad`` conditions, a guidance coefficient
    would only claim an effect this implementation cannot demonstrate.
    """

    def __init__(
        self,
        model: Lm,
        text_sampler: Sampler,
        audio_sampler: Sampler,
        batch_size: int = 1,
    ):
        self.model = model
        self.text_sampler = text_sampler
        self.audio_sampler = audio_sampler
        self.batch_size = batch_size

        cfg = model.cfg
        self.audio_delays = cfg.audio_delays
        self.max_delay = max(self.audio_delays)
        self.target_codebooks = cfg.target_codebooks
        self.audio_codebooks = cfg.audio_codebooks
        self.audio_padding_token = cfg.audio_padding_token
        self.text_start_token = cfg.text_out_vocab_size

        # A position is written at ``t`` and again at ``t + max_delay``, and is
        # last read at ``t + max_delay + 1``, so this many columns are live.
        self.window = self.max_delay + 2
        self.transformer_cache: list[LayerCache] = model.make_transformer_cache()
        self.depformer_cache: list[LayerCache] = model.make_depformer_cache()
        self.reset()

    def reset(self) -> None:
        """Return the schedule to the state it had before any audio arrived."""
        self.step_idx = 0
        self.gen_sequence = mx.full(
            shape=(self.batch_size, 1 + self.audio_codebooks, self.window),
            vals=UNGENERATED_TOKEN,
            dtype=mx.int32,
        )
        for cache in self.transformer_cache:
            cache.reset()
        for cache in self.depformer_cache:
            cache.reset()

    @property
    def text_frame_index(self) -> int:
        """The frame the next step will produce text for."""
        return self.step_idx

    def step(self, source_tokens: mx.array, condition: ConditionTensor | None = None) -> mx.array:
        """Advance one frame from the source tokens Mimi just produced.

        ``source_tokens`` is ``[B, source_codebooks]`` for the current frame.
        Returns the sampled text token as ``[B, 1]``; the audio for frame
        ``t - 2`` is then read with :meth:`last_audio_tokens`.
        """
        step_idx = self.step_idx
        column = step_idx % self.window
        # This column last held position ``step_idx - window``, which every
        # reader has already passed.
        self.gen_sequence[:, :, column] = UNGENERATED_TOKEN
        self.gen_sequence[:, 1 + self.target_codebooks :, column] = source_tokens

        if step_idx == 0:
            text_tokens = mx.full((self.batch_size, 1), self.text_start_token, dtype=mx.int32)
        else:
            text_tokens = self._read(0, step_idx - 1, "text")

        audio_tokens = [
            self._read(codebook + 1, step_idx - 1 - delay, f"audio codebook {codebook}")
            for codebook, delay in enumerate(self.audio_delays)
        ]

        text_token, generated = self.model.sample_step(
            text_tokens,
            audio_tokens,
            self.transformer_cache,
            self.depformer_cache,
            self.text_sampler,
            self.audio_sampler,
            condition=condition,
        )

        self.gen_sequence[:, 0, column] = text_token.squeeze(-1)
        for codebook, delay in enumerate(self.audio_delays[: self.target_codebooks]):
            position = step_idx - delay
            if position >= 0:
                self.gen_sequence[:, codebook + 1, position % self.window] = generated[
                    :, codebook, 0
                ]
        self.step_idx += 1
        return text_token

    def last_audio_tokens(self) -> mx.array | None:
        """The newest target frame whose eight codebooks are all present.

        That is frame ``t - 2``, so the first two steps of a session have no
        audio to return.
        """
        position = self.step_idx - 1 - self.max_delay
        if position < 0:
            return None
        tokens = self.gen_sequence[:, 1 : 1 + self.target_codebooks, position % self.window]
        if bool((tokens == self.audio_padding_token).any()):
            return None
        if bool((tokens == UNGENERATED_TOKEN).any()):
            raise ScheduleError(f"target audio frame {position} was never generated")
        return tokens

    @property
    def audio_frame_index(self) -> int:
        """The frame :meth:`last_audio_tokens` refers to after a step."""
        return self.step_idx - 1 - self.max_delay

    def _read(self, stream: int, position: int, what: str) -> mx.array:
        """Read one stream at one position, as a ``[B, 1]`` model input."""
        if position < 0:
            return mx.full((self.batch_size, 1), self.audio_padding_token, dtype=mx.int32)
        tokens = self.gen_sequence[:, stream, position % self.window][:, None]
        if bool((tokens == UNGENERATED_TOKEN).any()):
            raise ScheduleError(
                f"{what} at frame {position} was read at step {self.step_idx} "
                "before anything wrote it"
            )
        return tokens
