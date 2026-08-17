"""Command line entry point: ``python -m hibiki_mlx french.wav english.wav``."""

import sys

from .run_inference import main

sys.exit(main())
