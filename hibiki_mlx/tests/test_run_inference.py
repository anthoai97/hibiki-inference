"""Tests for offline-inference command-line helpers."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from hibiki_mlx.run_inference import (
    MemoryUsage,
    _format_step_metrics,
    _log,
    _stream_pcm,
    build_parser,
    main,
)
from hibiki_mlx.session import StepResult, StepTiming


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


class MetricsTests(unittest.TestCase):
    def test_formats_every_phase_and_memory_total_for_one_step(self) -> None:
        result = StepResult(
            text_frame_index=4,
            text_token=42,
            text=None,
            audio_frame_index=2,
            pcm=None,
            seconds_per_frame=0.08,
            timing=StepTiming(0.002, 0.003, 0.004, 0.001, 0.01),
        )
        memory = MemoryUsage(
            process_peak_rss_bytes=8 * 2**20,
            mlx_active_bytes=2 * 2**20,
            mlx_cache_bytes=1 * 2**20,
            mlx_peak_bytes=4 * 2**20,
        )

        self.assertEqual(
            _format_step_metrics(result, memory),
            "step=4 text_frame=4 audio_frame=2 phases: encode=2.0ms generate=3.0ms "
            "decode=4.0ms text=1.0ms total=10.0ms memory: mlx=3.0MiB "
            "(active=2.0MiB cache=1.0MiB peak_active=4.0MiB) process_peak_rss=8.0MiB",
        )

    def test_metrics_flag_is_opt_in(self) -> None:
        arguments = build_parser().parse_args(["--metrics", "input.wav"])

        self.assertTrue(arguments.metrics)

    def test_main_reports_metrics_for_each_result_and_at_run_end(self) -> None:
        result = StepResult(
            text_frame_index=4,
            text_token=42,
            text=None,
            audio_frame_index=2,
            pcm=None,
            seconds_per_frame=0.08,
            timing=StepTiming(0.002, 0.003, 0.004, 0.001, 0.01),
        )
        memory = MemoryUsage(
            process_peak_rss_bytes=8 * 2**20,
            mlx_active_bytes=2 * 2**20,
            mlx_cache_bytes=1 * 2**20,
            mlx_peak_bytes=4 * 2**20,
        )
        session = type("Session", (), {"frame_size": 3, "seconds_per_frame": 0.08})()
        with (
            patch("hibiki_mlx.run_inference.mx.random.seed"),
            patch("hibiki_mlx.run_inference.mx.reset_peak_memory"),
            patch("hibiki_mlx.run_inference.load_model"),
            patch("hibiki_mlx.run_inference.Sampler"),
            patch("hibiki_mlx.run_inference.InferenceSession", return_value=session) as session_type,
            patch("hibiki_mlx.run_inference.read_pcm", return_value=np.zeros(3, dtype=np.float32)),
            patch("hibiki_mlx.run_inference._stream_pcm", return_value=iter([result])),
            patch("hibiki_mlx.run_inference._memory_usage", side_effect=[memory, memory]),
            patch("hibiki_mlx.run_inference._log") as log,
        ):
            self.assertEqual(main(["--metrics", "--no-warmup", "input.wav"]), 0)

        self.assertTrue(session_type.call_args.kwargs["measure_timing"])
        messages = [call.args[0] for call in log.call_args_list]
        self.assertIn(_format_step_metrics(result, memory), messages)
        self.assertTrue(any(message.startswith("metrics totals: steps=1") for message in messages))
