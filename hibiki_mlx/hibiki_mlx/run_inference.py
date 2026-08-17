"""Translate a French audio file into English text and English audio.

    python -m hibiki_mlx french.wav english.wav

This is the offline counterpart of a live session: the file is fed to the same
streaming session one 80 ms frame at a time, in order, with no lookahead. The
only thing it adds is that the whole input happens to be available up front, so
it can report how far ahead of real time the run stayed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, read_pcm, write_wav
from .download import DEFAULT_ARTIFACT_DIRECTORY
from .inference import load_model
from .session import DEFAULT_CONDITION, InferenceSession, StepResult
from .sampling import Sampler

DEFAULT_SEED = 299792458


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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    mx.random.seed(arguments.seed)

    _log(f"loading the artifact bundle from {arguments.artifacts}")
    model = load_model(artifact_directory=arguments.artifacts)

    session = InferenceSession(
        model,
        condition=arguments.condition,
        text_sampler=Sampler(temp=arguments.temp, top_k=arguments.text_top_k),
        audio_sampler=Sampler(temp=arguments.temp, top_k=arguments.audio_top_k),
    )
    if not arguments.no_warmup:
        _log("warming up")
        session.warmup()

    _log(f"reading {arguments.infile}")
    pcm = read_pcm(arguments.infile)
    frames = len(pcm) // session.frame_size
    _log(f"{len(pcm) / SAMPLE_RATE:.1f}s of audio, {frames} frames to translate")

    started = time.time()
    results = session.push_pcm(pcm)
    _echo(results)
    results += _echo(session.finish())
    elapsed = time.time() - started
    print()

    steps = len(results)
    real_time = 1.0 / session.seconds_per_frame
    _log(
        f"{steps} steps in {elapsed:.1f}s: {steps / elapsed:.1f} steps/s, "
        f"{real_time:.1f} needed for real time"
    )

    out_pcm = [result.pcm for result in results if result.pcm is not None]
    if arguments.outfile:
        if not out_pcm:
            _log("no complete audio frame was produced, so no file was written")
            return 1
        _log(f"writing {len(out_pcm)} frames to {arguments.outfile}")
        write_wav(arguments.outfile, np.concatenate(out_pcm))
    return 0


def _echo(results: list[StepResult]) -> list[StepResult]:
    for result in results:
        if result.text:
            print(result.text, end="", flush=True)
    return results


def _log(message: str) -> None:
    print(f"[hibiki] {message}", file=sys.stderr, flush=True)
