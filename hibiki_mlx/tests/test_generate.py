"""Tests for the delayed-stream schedule.

The schedule is exercised against a stand-in model that records what it was fed
and returns predictable tokens, so the delay arithmetic is checked without the
released weights.
"""

from __future__ import annotations

import unittest

import mlx.core as mx

from hibiki_mlx.generate import UNGENERATED_TOKEN, LmGen, ScheduleError
from hibiki_mlx.models.lm import LmConfig
from hibiki_mlx.sampling import Sampler

from .test_lm_config import released_config

PADDING = 2048
BOS = 48000


def text_token_for(step: int) -> int:
    return 1000 + step


def audio_token_for(step: int, codebook: int) -> int:
    return 10 * step + codebook


def source_token_for(step: int, codebook: int) -> int:
    return 500 + 10 * step + codebook


def delay_of(codebook: int) -> int:
    """Codebook 0 of a stream is the semantic one and carries no delay."""
    return 0 if codebook == 0 else 2


def target_of_frame(frame: int, codebook: int) -> int:
    """The token that belongs to ``frame``.

    A delayed codebook is sampled ``delay`` steps after the frame it belongs
    to, so frame ``f``'s codebook 3 is whatever step ``f + 2`` produced.
    """
    return audio_token_for(frame + delay_of(codebook), codebook)


class FakeLm:
    """Stands in for the generator: records its inputs, returns known tokens."""

    def __init__(self, cfg: LmConfig):
        self.cfg = cfg
        self.calls: list[tuple[int, list[int]]] = []

    def make_transformer_cache(self) -> list:
        return []

    def make_depformer_cache(self) -> list:
        return []

    def sample_step(
        self,
        text_token_ids,
        audio_token_ids,
        transformer_cache,
        depformer_cache,
        text_sampler,
        audio_sampler,
        condition=None,
    ):
        step = len(self.calls)
        self.calls.append(
            (
                int(text_token_ids.squeeze().item()),
                [int(token.squeeze().item()) for token in audio_token_ids],
            )
        )
        text = mx.array([[text_token_for(step)]], dtype=mx.int32)
        audio = mx.array(
            [[[audio_token_for(step, codebook)] for codebook in range(8)]],
            dtype=mx.int32,
        )
        return text, audio


def make_generator() -> tuple[LmGen, FakeLm]:
    model = FakeLm(LmConfig.from_config_dict(released_config()))
    return LmGen(model, Sampler(temp=0), Sampler(temp=0)), model


def push(generator: LmGen, step: int) -> None:
    generator.step(
        mx.array([[source_token_for(step, codebook) for codebook in range(8)]], dtype=mx.int32)
    )


class ScheduleTests(unittest.TestCase):
    def test_feeds_the_start_token_and_padding_on_the_first_step(self) -> None:
        generator, model = make_generator()

        push(generator, 0)

        text, audio = model.calls[0]
        self.assertEqual(text, BOS)
        self.assertEqual(audio, [PADDING] * 16)

    def test_feeds_the_previous_text_token_from_the_second_step_on(self) -> None:
        generator, model = make_generator()

        push(generator, 0)
        push(generator, 1)

        self.assertEqual(model.calls[1][0], text_token_for(0))

    def test_delays_seven_of_the_eight_codebooks_by_two_frames(self) -> None:
        generator, model = make_generator()

        for step in range(4):
            push(generator, step)

        _, audio = model.calls[3]
        # Codebook 0 of each stream has no delay, so step 3 is fed frame 2;
        # the delayed ones are fed frame 0.
        self.assertEqual(audio[8], source_token_for(2, 0))
        self.assertEqual(audio[9:], [source_token_for(0, cb) for cb in range(1, 8)])
        self.assertEqual(audio[0], target_of_frame(2, 0))
        self.assertEqual(audio[1:8], [target_of_frame(0, cb) for cb in range(1, 8)])
        # Frame 0's delayed codebooks were sampled at step 2, so every target
        # column fed back at step 3 came out of step 2.
        self.assertEqual(audio[:8], [audio_token_for(2, cb) for cb in range(8)])

    def test_pads_the_delayed_codebooks_until_their_frames_exist(self) -> None:
        generator, model = make_generator()

        push(generator, 0)
        push(generator, 1)
        push(generator, 2)

        _, audio = model.calls[2]
        self.assertEqual(audio[0], audio_token_for(1, 0))
        self.assertEqual(audio[1:8], [PADDING] * 7)

    def test_holds_audio_back_until_every_codebook_of_a_frame_is_ready(self) -> None:
        generator, _ = make_generator()

        push(generator, 0)
        self.assertIsNone(generator.last_audio_tokens())
        push(generator, 1)
        self.assertIsNone(generator.last_audio_tokens())
        push(generator, 2)

        tokens = generator.last_audio_tokens()
        self.assertIsNotNone(tokens)
        self.assertEqual(generator.audio_frame_index, 0)
        self.assertEqual(
            tokens.squeeze().tolist(),
            [target_of_frame(0, codebook) for codebook in range(8)],
        )

    def test_returns_audio_two_frames_behind_the_text(self) -> None:
        generator, _ = make_generator()

        for step in range(12):
            self.assertEqual(generator.text_frame_index, step)
            push(generator, step)
            if step >= 2:
                self.assertEqual(generator.audio_frame_index, step - 2)
                self.assertEqual(
                    generator.last_audio_tokens().squeeze().tolist(),
                    [target_of_frame(step - 2, codebook) for codebook in range(8)],
                )

    def test_keeps_the_delays_exact_well_past_the_buffered_window(self) -> None:
        generator, model = make_generator()

        for step in range(200):
            push(generator, step)

        for step in range(3, 200):
            _, audio = model.calls[step]
            self.assertEqual(audio[8], source_token_for(step - 1, 0), f"step {step}")
            self.assertEqual(audio[9], source_token_for(step - 3, 1), f"step {step}")
            self.assertEqual(audio[0], target_of_frame(step - 1, 0), f"step {step}")
            self.assertEqual(audio[1], target_of_frame(step - 3, 1), f"step {step}")

        for frame in range(190):
            self.assertEqual(
                model.calls[frame][0], text_token_for(frame - 1) if frame else BOS
            )

    def test_reset_returns_the_session_to_its_first_step(self) -> None:
        generator, model = make_generator()
        for step in range(5):
            push(generator, step)

        generator.reset()
        push(generator, 0)

        self.assertEqual(generator.text_frame_index, 1)
        self.assertIsNone(generator.last_audio_tokens())
        self.assertEqual(model.calls[-1][0], BOS)

    def test_refuses_to_read_a_position_nothing_wrote(self) -> None:
        generator, _ = make_generator()
        push(generator, 0)
        push(generator, 1)
        # Blank out the source stream the next step is about to read.
        generator.gen_sequence[:, 9:, 1 % generator.window] = UNGENERATED_TOKEN

        with self.assertRaises(ScheduleError):
            push(generator, 2)
