"""Allow ``python -m hibiki_mlx`` from the repository root during development."""

from __future__ import annotations

import sys

from .hibiki_mlx.run_inference import main


sys.exit(main())
