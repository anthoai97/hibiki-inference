# hibiki-mlx

Clean local MLX inference for Hibiki 1B French-to-English speech translation.

This package is an independent MLX reimplementation. It loads the released
Hibiki artifacts directly and does not depend on, wrap, or invoke `moshi_mlx`,
`rustymimi`, or `moshi-swift`.

> **Status: artifact loading only.** Streaming translation is not implemented
> yet. What works today is resolving and verifying an artifact bundle, which is
> the gate every later inference stage sits behind. See
> [issue #1](https://github.com/anthoai97/hibiki-inference/issues/1) for scope.

## Requirements

- Apple-silicon macOS 14 or newer
- Python 3.13 or 3.14
- MLX 0.32.x (development, CI, fixtures, and performance reports pin 0.32.0)

MLX publishes Apple-silicon macOS wheels only, so installing on another platform
fails at install time rather than at first inference.

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/), and `uv.lock`
pins the versions used for development, fixtures, and performance reports.
Repository development commands run in the `hibiki` Conda environment, invoking
`uv` from there:

```sh
conda run -n hibiki uv sync --group dev
```

## Artifact bundles

The model weights are **not** bundled: they are about 4 GB and are licensed
separately (CC-BY 4.0 — see [`NOTICE`](NOTICE)). A bundle is the configuration,
the Hibiki weights, the Mimi weights, and the SentencePiece tokenizer from **one**
model revision, treated as a single verified unit.

```python
from hibiki_mlx import load_artifact_bundle

# Pinned Hugging Face revision (downloads into the Hugging Face cache).
bundle = load_artifact_bundle("kyutai/hibiki-1b-mlx-bf16")

# A prepared local directory, with no network access at all.
bundle = load_artifact_bundle("/path/to/hibiki-1b-mlx-bf16", offline=True)
```

Loading verifies, before any inference code can use the bundle:

- all four required artifacts are present, and nothing unexpected sits alongside them;
- `config.json` names exactly the weight and tokenizer files it is loaded with,
  so artifacts from different revisions cannot be mixed;
- every configuration invariant this implementation depends on — stream layout,
  codebook split, delay schedule, attention context, cardinalities, layer counts,
  and conditioner labels;
- every parameter name, dtype, and tensor shape in both safetensors files, checked
  against the architecture that `config.json` describes.

Only headers are read, so a 3.6 GB artifact is validated without loading weights.

### Safe mode and `allow_unsafe`

By default the loader accepts only the pinned revision
`b3d6291f3dcf7954e1a502e4d66f32e3556f17ae`, and checks each artifact's size and
SHA-256 against built-in values.

`allow_unsafe=True` relaxes **only** revision and hash trust. File, configuration,
parameter-name, and shape validation always run — so an untrusted bundle is still
rejected unless it structurally matches this implementation.

`offline=True` makes no network request; a bundle that is not already cached
locally fails rather than being fetched.

### Errors

All failures are public and distinguishable:

| Error | Raised when |
| --- | --- |
| `ArtifactResolutionError` | the bundle cannot be located: missing directory, malformed repository id, or `offline=True` with nothing cached |
| `ArtifactContentError` | the bundle's contents are wrong: a required artifact is missing, an unexpected file is present, or `config.json` disagrees with the filenames next to it |
| `UntrustedArtifactError` | safe mode rejected the revision, a file size, or a hash. The only error `allow_unsafe=True` suppresses |
| `IncompatibleArtifactError` | the artifacts do not match this implementation: a configuration invariant, parameter name, dtype, or tensor shape is wrong |

Each derives from `ArtifactError`, and in turn from `HibikiError`.

## Tests

Quick tests need neither network access nor model weights:

```sh
conda run -n hibiki uv run pytest
```

Model-backed tests are opt-in and require a real bundle on disk. They default to
`artifacts/hibiki-1b-mlx-bf16/` in this repository:

```sh
HIBIKI_MLX_TEST_BUNDLE=/path/to/bundle conda run -n hibiki uv run pytest -m model_backed
```

One of them additionally downloads the pinned revision from the Hub, so it is
gated behind its own variable:

```sh
HIBIKI_MLX_TEST_HUB=1 conda run -n hibiki uv run pytest -m model_backed
```

Type checking and linting:

```sh
conda run -n hibiki uv run mypy
conda run -n hibiki uv run ruff check
```

## License

Package code is Apache 2.0 ([`LICENSE`](LICENSE)). The model artifacts it loads
are CC-BY 4.0 and are not redistributed here ([`NOTICE`](NOTICE)).
