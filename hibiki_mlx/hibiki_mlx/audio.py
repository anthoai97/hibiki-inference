"""Reading and writing the 24 kHz mono PCM the model works in.

The standard library covers 16-bit WAV, which is what the checked-in assets are.
Anything else - a different rate, more channels, another container - is read
through `sphn` when it is installed, and refused with an explicit message when
it is not, rather than silently mis-timed.
"""

from __future__ import annotations

import queue
import threading
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


class AudioError(RuntimeError):
    """Raised when audio cannot be read at the rate and layout the model needs."""


class PlaybackError(RuntimeError):
    """Raised when the translated audio cannot be played."""


class PlaybackStream:
    """Play decoded PCM on a worker without blocking the inference loop.

    The bounded queue retains at most a short amount of output if generation
    runs ahead of the sound device. That keeps application I/O outside MLX and
    prevents an unlimited audio backlog.
    """

    _STOP = object()

    def __init__(self, max_chunks: int = 25) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(max_chunks)
        self._ready = threading.Event()
        self._aborted = threading.Event()
        self._failure: PlaybackError | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._play, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self.abort()
            raise PlaybackError("timed out starting audio playback")
        self._raise_if_failed()

    def play(self, pcm: np.ndarray) -> None:
        """Queue one decoded PCM chunk for playback."""
        if self._closed:
            raise PlaybackError("audio playback is already closed")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1, 1).copy()
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(samples, timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        """Finish playing queued audio before releasing the sound device."""
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        while self._thread.is_alive() and self._failure is None:
            try:
                self._queue.put(self._STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_failed()

    def abort(self) -> None:
        """Stop playback without waiting for queued audio to drain."""
        self._closed = True
        self._aborted.set()
        self._thread.join()

    def _play(self) -> None:
        try:
            import sounddevice

            with sounddevice.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            ) as output:
                self._ready.set()
                while not self._aborted.is_set():
                    try:
                        chunk = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if chunk is self._STOP:
                        return
                    output.write(chunk)
        except ImportError as error:
            self._failure = PlaybackError(
                "audio playback requires the sounddevice package"
            )
            self._failure.__cause__ = error
        except Exception as error:
            self._failure = PlaybackError(f"could not play audio: {error}")
        finally:
            self._ready.set()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure


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
