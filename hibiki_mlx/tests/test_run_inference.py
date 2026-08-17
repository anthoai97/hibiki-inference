"""Tests for offline-inference command-line helpers."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from hibiki_mlx.run_inference import _log


class LoggingTests(unittest.TestCase):
    def test_writes_prefixed_messages_to_standard_error(self) -> None:
        stderr = StringIO()

        with redirect_stderr(stderr):
            _log("loading the artifact bundle")

        self.assertEqual(stderr.getvalue(), "[hibiki] loading the artifact bundle\n")
