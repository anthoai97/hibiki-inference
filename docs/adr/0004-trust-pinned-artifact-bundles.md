---
status: accepted
---

# Trust only verified artifact bundles by default

Safe loading accepts the exact pinned Hibiki BF16 revision and verifies the four-file bundle with built-in hashes and strict configuration and weight checks, whether resolved from Hugging Face or a local directory. Other revisions require an explicit unsafe opt-in but still undergo structural validation, and offline loading is a hard no-network mode; this favors reproducibility and early failure over permissive checkpoint compatibility.
