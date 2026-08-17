"""One streaming translation run over a loaded model.

The session owns everything that changes while French speech is being
translated: the codec's caches and convolution state, the delayed-stream
schedule, the Transformer caches, the text decoder, and the frame counters. The
loaded model itself stays immutable and can start another session after
:meth:`reset`.

Text and audio leave a step on different positions of the model timeline. The
text belongs to frame ``t``; the audio that becomes complete during the same
call belongs to frame ``t - 2``, because seven of the eight codebooks are
delayed. :class:`StepResult` keeps both indices rather than pretending the two
are synchronous.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .generate import LmGen
from .inference import LoadedModel
from .modules.conditioner import ConditionTensor
from .sampling import Sampler
from .text import TextDecoder

DEFAULT_TEXT_SAMPLER = Sampler(temp=0.8, top_k=25)
DEFAULT_AUDIO_SAMPLER = Sampler(temp=0.8, top_k=250)
DEFAULT_CONDITION = "very_good"

# Frames of silence pushed after the input ends, so the delayed codebooks of the
# last real frames can still be completed. This is an explicit fallback, not
# learned end-of-stream behaviour.
SILENCE_TAIL_FRAMES = 6


@dataclass(frozen=True)
class StepTiming:
    """Wall-clock time spent in each part of one generation step."""

    source_encode_seconds: float
    generation_seconds: float
    target_decode_seconds: float
    text_decode_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class StepResult:
    """What one generation step produced."""

    text_frame_index: int
    text_token: int
    text: str | None
    audio_frame_index: int | None
    pcm: np.ndarray | None
    seconds_per_frame: float
    timing: StepTiming | None = None

    @property
    def text_time(self) -> float:
        """The text's Model time, in seconds."""
        return self.text_frame_index * self.seconds_per_frame

    @property
    def audio_time(self) -> float | None:
        """The target audio frame's Model time, in seconds."""
        if self.audio_frame_index is None:
            return None
        return self.audio_frame_index * self.seconds_per_frame


class InferenceSession:
    """Push French PCM in; get English text and English PCM out."""

    def __init__(
        self,
        model: LoadedModel,
        *,
        condition: str | None = DEFAULT_CONDITION,
        text_sampler: Sampler = DEFAULT_TEXT_SAMPLER,
        audio_sampler: Sampler = DEFAULT_AUDIO_SAMPLER,
        measure_timing: bool = False,
    ):
        self.model = model
        self.mimi = model.mimi
        self.frame_size = model.mimi.cfg.frame_size
        self.seconds_per_frame = 1.0 / model.mimi.cfg.frame_rate
        self.measure_timing = measure_timing
        self.generator = LmGen(model.lm, text_sampler, audio_sampler)
        # Both the tokenizer and the no-text id come from the same bundle, so
        # text ids cannot be paired with weights from another revision.
        self.text_decoder = TextDecoder(
            model.tokenizer,
            no_text_token=model.lm_config.text_padding_token,
        )
        self.condition: ConditionTensor | None = None
        if condition is not None:
            provider = model.lm.condition_provider
            if provider is None:
                raise ValueError(f"the bundle has no conditioner to set to {condition!r}")
            self.condition = provider.condition_tensor("description", condition)
        self._encoder_cache = self.mimi.make_encoder_cache()
        self._decoder_cache = self.mimi.make_decoder_cache()
        # The codec's streaming convolutions keep their state inside the loaded
        # model, so a new session must clear it or it inherits the tail of
        # whatever ran last.
        self.reset()

    @property
    def text(self) -> str:
        """Everything translated so far in this session."""
        return self.text_decoder.text

    def reset(self) -> None:
        """Drop all streaming state so the loaded model can be reused."""
        self.generator.reset()
        self.text_decoder.reset()
        self.mimi.reset_state()
        for cache in self._encoder_cache:
            cache.reset()
        for cache in self._decoder_cache:
            cache.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._finished = False

    def warmup(self) -> None:
        """Run one frame through every fixed-shape path, then reset.

        Cold compilation costs far more than the 80 ms frame budget, so it must
        happen before a caller starts measuring or streaming.
        """
        self.push_pcm(np.zeros(self.frame_size, dtype=np.float32))
        self.reset()

    def push_pcm(self, pcm: np.ndarray) -> list[StepResult]:
        """Translate as much of the buffered audio as makes whole frames.

        Chunks may be any length; whatever does not fill a frame is kept for the
        next call.
        """
        if self._finished:
            raise RuntimeError("this session has been finished; call reset() to reuse it")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, samples])

        results = []
        while len(self._pending) >= self.frame_size:
            frame, self._pending = (
                self._pending[: self.frame_size],
                self._pending[self.frame_size :],
            )
            results.extend(self._step_frame(frame))
        return results

    def finish(self) -> list[StepResult]:
        """Pad the leftover PCM chunk and drain the delayed audio with silence.

        The tail also lets the translation catch up: Hibiki lags behind the
        French it is translating, so stopping at the last source frame cuts the
        final words off mid-sentence.
        """
        if self._finished:
            return []
        results = []
        if len(self._pending) > 0:
            tail = np.zeros(self.frame_size, dtype=np.float32)
            tail[: len(self._pending)] = self._pending
            self._pending = np.zeros(0, dtype=np.float32)
            results.extend(self._step_frame(tail))
        silence = np.zeros(self.frame_size, dtype=np.float32)
        for _ in range(SILENCE_TAIL_FRAMES):
            results.extend(self._step_frame(silence))
        self._finished = True
        return results

    def _step_frame(self, frame: np.ndarray) -> list[StepResult]:
        """Encode one source frame and run every generation step it yields."""
        started = time.perf_counter() if self.measure_timing else None
        codes = self.mimi.encode_step(mx.array(frame)[None, None, :], self._encoder_cache)
        source_encode_seconds = (
            (time.perf_counter() - started) / codes.shape[-1] if started is not None else None
        )
        return [
            self._generate(codes[:, :, index], source_encode_seconds)
            for index in range(codes.shape[-1])
        ]

    def _generate(
        self, source_tokens: mx.array, source_encode_seconds: float | None = None
    ) -> StepResult:
        text_frame_index = self.generator.text_frame_index
        started = time.perf_counter() if source_encode_seconds is not None else None
        text_token = self.generator.step(source_tokens.astype(mx.int32), self.condition)
        audio_tokens = self.generator.last_audio_tokens()
        if started is not None:
            if audio_tokens is None:
                mx.eval(text_token)
            else:
                mx.eval(text_token, audio_tokens)
            generation_seconds = time.perf_counter() - started

        pcm = None
        audio_frame_index = None
        decode_started = time.perf_counter() if started is not None else None
        if audio_tokens is not None:
            audio_frame_index = self.generator.audio_frame_index
            decoded = self.mimi.decode_step(audio_tokens[:, :, None], self._decoder_cache)
            mx.eval(decoded)
            pcm = np.array(decoded[0, 0], dtype=np.float32)
        target_decode_seconds = (
            time.perf_counter() - decode_started if decode_started is not None else None
        )

        text_started = time.perf_counter() if started is not None else None
        token = int(text_token.squeeze().item())
        text = self.text_decoder.push(token)
        text_decode_seconds = time.perf_counter() - text_started if text_started is not None else None
        timing = None
        if source_encode_seconds is not None:
            assert generation_seconds is not None
            assert target_decode_seconds is not None
            assert text_decode_seconds is not None
            timing = StepTiming(
                source_encode_seconds=source_encode_seconds,
                generation_seconds=generation_seconds,
                target_decode_seconds=target_decode_seconds,
                text_decode_seconds=text_decode_seconds,
                total_seconds=(
                    source_encode_seconds
                    + generation_seconds
                    + target_decode_seconds
                    + text_decode_seconds
                ),
            )
        return StepResult(
            text_frame_index=text_frame_index,
            text_token=token,
            text=text,
            audio_frame_index=audio_frame_index,
            pcm=pcm,
            seconds_per_frame=self.seconds_per_frame,
            timing=timing,
        )
