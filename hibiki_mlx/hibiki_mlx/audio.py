"""Reading and writing the 24 kHz mono PCM the model works in.

The standard library covers 16-bit WAV, which is what the checked-in assets are.
Anything else - a different rate, more channels, another container - is read
through `sphn` when it is installed, and refused with an explicit message when
it is not, rather than silently mis-timed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


class AudioError(RuntimeError):
    """Raised when audio cannot be read at the rate and layout the model needs."""


def read_pcm(path: str | Path) -> np.ndarray:
    """Read one audio file as mono float32 PCM at 24 kHz."""
    path = Path(path)
    native = _read_wav_24k_mono(path)
    if native is not None:
        return native

    try:
        import sphn
    except ImportError as error:
        raise AudioError(
            f"{path} is not 16-bit mono WAV at {SAMPLE_RATE} Hz. Convert it first, "
            "or install sphn to have it converted on the fly."
        ) from error
    pcm, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    return np.asarray(pcm, dtype=np.float32)[0]


def write_wav(path: str | Path, pcm: np.ndarray) -> None:
    """Write mono float32 PCM as a 16-bit WAV at 24 kHz."""
    samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
    # Round to the nearest step of the same scale the reader divides by, so a
    # round trip costs half a step rather than a step and a half.
    samples = np.clip(np.round(samples * 32768.0), -32768.0, 32767.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.astype("<i2").tobytes())


def _read_wav_24k_mono(path: Path) -> np.ndarray | None:
    """Read a 16-bit mono 24 kHz WAV, or return None if it is something else."""
    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getsampwidth() != 2
                or handle.getframerate() != SAMPLE_RATE
            ):
                return None
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error):
        return None
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
