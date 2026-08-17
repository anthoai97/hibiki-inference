---
status: accepted
---

# Keep the Python implementation in a separate package area

The first implementation will be the `hibiki_mlx` distribution under the top-level `hibiki_mlx/` area, imported as `hibiki_mlx`; a future native Swift implementation will remain separate and share artifact, scheduling, timestamp, and parity contracts rather than reuse Python through a bridge. Version one targets Apple-silicon macOS 14 or newer with Python 3.13–3.14 and MLX 0.32, uses `uv`, and is installable locally without being published to PyPI.
