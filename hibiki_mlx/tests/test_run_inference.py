"""Tests for offline-inference command-line helpers."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

import numpy as np

from hibiki_mlx.run_inference import _log, _stream_pcm
from hibiki_mlx.session import StepResult


class LoggingTests(unittest.TestCase):
    def test_writes_prefixed_messages_to_standard_error(self) -> None:
        stderr = StringIO()

        with redirect_stderr(stderr):
            _log("loading the artifact bundle")

        self.assertEqual(stderr.getvalue(), "[hibiki] loading the artifact bundle\n")


def step(text: str | None) -> StepResult:
    return StepResult(
        text_frame_index=0,
        text_token=0,
        text=text,
        audio_frame_index=None,
        pcm=None,
        seconds_per_frame=0.08,
    )


class StreamingSession:
    frame_size = 3

    def __init__(self) -> None:
        self.chunks: list[list[float]] = []
        self.finished = False

    def push_pcm(self, pcm: np.ndarray) -> list[StepResult]:
        self.chunks.append(pcm.tolist())
        return [step(str(len(self.chunks)))]

    def finish(self) -> list[StepResult]:
        self.finished = True
        return [step("tail")]


class StreamingTests(unittest.TestCase):
    def test_yields_results_after_each_source_frame_before_finishing(self) -> None:
        session = StreamingSession()
        results = _stream_pcm(session, np.arange(7, dtype=np.float32))

        self.assertEqual(next(results).text, "1")
        self.assertEqual(session.chunks, [[0.0, 1.0, 2.0]])
        self.assertFalse(session.finished)

        self.assertEqual(next(results).text, "2")
        self.assertEqual(next(results).text, "3")
        self.assertEqual(next(results).text, "tail")
        with self.assertRaises(StopIteration):
            next(results)

        self.assertEqual(session.chunks, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0]])
        self.assertTrue(session.finished)
