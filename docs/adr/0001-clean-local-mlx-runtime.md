---
status: accepted
---

# Implement the production runtime locally in MLX

The production package will load the released Hibiki and Mimi artifacts directly and implement the codec, generator, scheduler, caches, and sampling locally in MLX. Pinned Moshi and Hibiki implementations may be used in isolated developer tooling to create parity fixtures, but they will not be imported, wrapped, copied into, or installed by the production runtime; this costs more initial implementation work but avoids inheriting obsolete runtime constraints and gives a later native Swift implementation a clean shared compatibility contract.
