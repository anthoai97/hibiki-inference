"""Tests for reading and writing the model's 24 kHz mono PCM."""

from __future__ import annotations

import sys
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hibiki_mlx.audio import SAMPLE_RATE, PlaybackStream, read_pcm, write_wav
from hibiki_mlx.audio import _read_wav_24k_mono as read_wav_24k_mono


def tone(seconds: float = 0.1, rate: int = SAMPLE_RATE) -> np.ndarray:
    times = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)


class RoundTripTests(unittest.TestCase):
    def test_reads_back_what_it_wrote(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            original = tone()

            write_wav(path, original)
            restored = read_pcm(path)

            self.assertEqual(restored.shape, original.shape)
            self.assertEqual(restored.dtype, np.float32)
            # 16-bit quantisation is the only loss.
            self.assertLess(float(np.abs(restored - original).max()), 1 / 32767)

    def test_writes_the_rate_and_layout_the_model_works_in(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"

            write_wav(path, tone())

            with wave.open(str(path)) as handle:
                self.assertEqual(handle.getframerate(), SAMPLE_RATE)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)

    def test_clips_instead_of_wrapping_around(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "loud.wav"

            write_wav(path, np.array([2.0, -2.0], dtype=np.float32))

            self.assertTrue(np.all(np.abs(read_pcm(path)) <= 1.0))


class WavRateTests(unittest.TestCase):
    def test_declines_a_file_that_is_not_at_the_model_rate(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fast.wav"
            samples = (tone(rate=48000) * 32767).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(48000)
                handle.writeframes(samples.tobytes())

            # Resampling belongs to the optional reader, not to this one.
            self.assertIsNone(read_wav_24k_mono(path))

    def test_declines_a_file_that_is_not_mono(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            samples = (tone() * 32767).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(np.repeat(samples, 2).tobytes())

            self.assertIsNone(read_wav_24k_mono(path))


class FakeOutputStream:
    instances: list["FakeOutputStream"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.writes: list[np.ndarray] = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def write(self, pcm: np.ndarray) -> None:
        self.writes.append(pcm.copy())


class PlaybackStreamTests(unittest.TestCase):
    def test_plays_float_pcm_at_the_model_rate(self) -> None:
        FakeOutputStream.instances.clear()
        sounddevice = SimpleNamespace(OutputStream=FakeOutputStream)
        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            playback = PlaybackStream()
            playback.play(np.array([0.25, -0.5], dtype=np.float32))
            playback.close()

        self.assertEqual(len(FakeOutputStream.instances), 1)
        output = FakeOutputStream.instances[0]
        self.assertEqual(
            output.kwargs,
            {"samplerate": SAMPLE_RATE, "channels": 1, "dtype": "float32"},
        )
        np.testing.assert_array_equal(
            output.writes,
            [np.array([[0.25], [-0.5]], dtype=np.float32)],
        )
