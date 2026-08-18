"""Translate a French audio file into English text and English audio.

    python -m hibiki_mlx french.wav english.wav

This is the offline counterpart of a live session: the file is fed to the same
streaming session one 80 ms frame at a time, in order, with no lookahead. The
only thing it adds is that the whole input happens to be available up front, so
it can report how far ahead of real time the run stayed.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, PlaybackStream, read_pcm, write_wav
from .download import DEFAULT_ARTIFACT_DIRECTORY
from .inference import load_model
from .session import DEFAULT_CONDITION, InferenceSession, StepResult
from .sampling import Sampler

DEFAULT_SEED = 299792458


@dataclass(frozen=True)
class MemoryUsage:
    """Process and MLX memory observed at one point during a run."""

    process_peak_rss_bytes: int
    mlx_active_bytes: int
    mlx_cache_bytes: int
    mlx_peak_bytes: int

    @property
    def mlx_total_bytes(self) -> int:
        return self.mlx_active_bytes + self.mlx_cache_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hibiki_mlx", description=__doc__)
    parser.add_argument("infile", help="French audio to translate")
    parser.add_argument("outfile", nargs="?", help="where to write the English audio")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="directory holding the artifact bundle",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--text-top-k", type=int, default=25)
    parser.add_argument("--audio-top-k", type=int, default=250)
    parser.add_argument(
        "--condition",
        default=DEFAULT_CONDITION,
        help="quality label fed to the conditioner",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the warmup frame, leaving cold compilation in the first steps",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="play decoded English audio as it becomes available",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="report phase timings and memory for every generation step",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    mx.random.seed(arguments.seed)

    run_started = time.perf_counter()
    _log(f"loading the artifact bundle from {arguments.artifacts}")
    phase_started = time.perf_counter()
    model = load_model(artifact_directory=arguments.artifacts)
    load_seconds = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    session = InferenceSession(
        model,
        condition=arguments.condition,
        text_sampler=Sampler(temp=arguments.temp, top_k=arguments.text_top_k),
        audio_sampler=Sampler(temp=arguments.temp, top_k=arguments.audio_top_k),
        measure_timing=arguments.metrics,
    )
    session_seconds = time.perf_counter() - phase_started
    warmup_seconds = 0.0
    if not arguments.no_warmup:
        _log("warming up")
        phase_started = time.perf_counter()
        session.warmup()
        warmup_seconds = time.perf_counter() - phase_started

    _log(f"reading {arguments.infile}")
    phase_started = time.perf_counter()
    pcm = read_pcm(arguments.infile)
    read_seconds = time.perf_counter() - phase_started
    frames = len(pcm) // session.frame_size
    _log(f"{len(pcm) / SAMPLE_RATE:.1f}s of audio, {frames} frames to translate")

    playback = None
    if arguments.play:
        _log("starting English audio playback")
        playback = PlaybackStream()

    if arguments.metrics:
        mx.reset_peak_memory()

    started = time.perf_counter()
    try:
        results = []
        for result in _stream_pcm(session, pcm):
            _echo([result])
            if playback is not None and result.pcm is not None:
                playback.play(result.pcm)
            if arguments.metrics:
                _log(_format_step_metrics(result, _memory_usage()))
            results.append(result)
    except BaseException:
        if playback is not None:
            playback.abort()
        raise
    else:
        if playback is not None:
            playback.close()
    elapsed = time.perf_counter() - started
    print()

    steps = len(results)
    real_time = 1.0 / session.seconds_per_frame
    _log(
        f"{steps} steps in {elapsed:.1f}s: {steps / elapsed:.1f} steps/s, "
        f"{real_time:.1f} needed for real time"
    )

    phase_started = time.perf_counter()
    out_pcm = [result.pcm for result in results if result.pcm is not None]
    if arguments.outfile:
        if not out_pcm:
            _log("no complete audio frame was produced, so no file was written")
            return 1
        _log(f"writing {len(out_pcm)} frames to {arguments.outfile}")
        write_wav(arguments.outfile, np.concatenate(out_pcm))
    write_seconds = time.perf_counter() - phase_started
    if arguments.metrics:
        _log(
            _format_run_metrics(
                results,
                {
                    "load": load_seconds,
                    "session": session_seconds,
                    "warmup": warmup_seconds,
                    "read": read_seconds,
                    "translate": elapsed,
                    "write": write_seconds,
                    "run": time.perf_counter() - run_started,
                },
                _memory_usage(),
            )
        )
    return 0


def _stream_pcm(session: InferenceSession, pcm: np.ndarray) -> Iterator[StepResult]:
    """Yield each result as soon as one source frame has been translated."""
    for start in range(0, len(pcm), session.frame_size):
        yield from session.push_pcm(pcm[start : start + session.frame_size])
    yield from session.finish()


def _echo(results: list[StepResult]) -> list[StepResult]:
    for result in results:
        if result.text:
            print(result.text, end="", flush=True)
    return results


def _memory_usage() -> MemoryUsage:
    """Read memory counters without allocating tensors or synchronizing MLX."""
    process_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        process_peak_rss *= 1024
    return MemoryUsage(
        process_peak_rss_bytes=process_peak_rss,
        mlx_active_bytes=int(mx.get_active_memory()),
        mlx_cache_bytes=int(mx.get_cache_memory()),
        mlx_peak_bytes=int(mx.get_peak_memory()),
    )


def _format_step_metrics(result: StepResult, memory: MemoryUsage) -> str:
    """Format one monitor line for a completed generation step."""
    timing = result.timing
    if timing is None:
        raise ValueError("step timing is unavailable; create the session with measure_timing=True")
    audio_frame = "-" if result.audio_frame_index is None else str(result.audio_frame_index)
    return (
        f"step={result.text_frame_index} text_frame={result.text_frame_index} "
        f"audio_frame={audio_frame} phases: "
        f"encode={_milliseconds(timing.source_encode_seconds)} "
        f"generate={_milliseconds(timing.generation_seconds)} "
        f"decode={_milliseconds(timing.target_decode_seconds)} "
        f"text={_milliseconds(timing.text_decode_seconds)} "
        f"total={_milliseconds(timing.total_seconds)} memory: "
        f"mlx={_mebibytes(memory.mlx_total_bytes)} "
        f"(active={_mebibytes(memory.mlx_active_bytes)} "
        f"cache={_mebibytes(memory.mlx_cache_bytes)} "
        f"peak_active={_mebibytes(memory.mlx_peak_bytes)}) "
        f"process_peak_rss={_mebibytes(memory.process_peak_rss_bytes)}"
    )


def _format_run_metrics(
    results: list[StepResult], phase_seconds: dict[str, float], memory: MemoryUsage
) -> str:
    """Format run-wide phase timing and final memory totals."""
    timings = [result.timing for result in results if result.timing is not None]
    phases = " ".join(
        f"{name}={_milliseconds(seconds)}" for name, seconds in phase_seconds.items()
    )
    measured = sum(timing.total_seconds for timing in timings)
    return (
        f"metrics totals: steps={len(results)} measured_steps={_milliseconds(measured)} "
        f"phases: {phases} memory: mlx={_mebibytes(memory.mlx_total_bytes)} "
        f"(active={_mebibytes(memory.mlx_active_bytes)} "
        f"cache={_mebibytes(memory.mlx_cache_bytes)} "
        f"peak_active={_mebibytes(memory.mlx_peak_bytes)}) "
        f"process_peak_rss={_mebibytes(memory.process_peak_rss_bytes)}"
    )


def _milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def _mebibytes(bytes_count: int) -> str:
    return f"{bytes_count / 2**20:.1f}MiB"


def _log(message: str) -> None:
    print(f"[hibiki] {message}", file=sys.stderr, flush=True)
