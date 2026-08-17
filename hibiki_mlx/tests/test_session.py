"""Tests for what a generation step reports.

These need no weights: a step result is a plain record, and the model clock it
reports against comes from the codec's frame rate rather than a literal.
"""

from __future__ import annotations

import unittest

from hibiki_mlx.session import StepResult

# 12.5 frames per second, the released codec's clock.
FRAME = 1 / 12.5


def result(text_frame: int, audio_frame: int | None, seconds_per_frame: float = FRAME) -> StepResult:
    return StepResult(
        text_frame_index=text_frame,
        text_token=42,
        text=None,
        audio_frame_index=audio_frame,
        pcm=None,
        seconds_per_frame=seconds_per_frame,
    )


class ModelTimeTests(unittest.TestCase):
    def test_places_text_on_the_model_clock(self) -> None:
        self.assertAlmostEqual(result(0, None).text_time, 0.0)
        self.assertAlmostEqual(result(1, None).text_time, 0.08)
        self.assertAlmostEqual(result(25, 23).text_time, 2.0)

    def test_places_audio_two_frames_behind_the_text(self) -> None:
        step = result(25, 23)

        self.assertAlmostEqual(step.text_time, 2.0)
        self.assertAlmostEqual(step.audio_time, 1.84)

    def test_reports_no_audio_time_before_the_first_complete_frame(self) -> None:
        self.assertIsNone(result(1, None).audio_time)

    def test_follows_the_codec_clock_rather_than_a_fixed_eighty_milliseconds(self) -> None:
        # A codec at 25 Hz would put frame 10 at 0.4 s, not 0.8 s.
        step = result(10, 8, seconds_per_frame=1 / 25)

        self.assertAlmostEqual(step.text_time, 0.4)
        self.assertAlmostEqual(step.audio_time, 0.32)
