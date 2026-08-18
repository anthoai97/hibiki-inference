# HibikiCore

The native MLX Swift implementation of the S2S Edge inference path — artifact
bundle loading, the Mimi codec, and the Hibiki generator. It is a separate
Swift package (not app code) so the same modules build into the **Hibiki Edge**
iOS app *and* run under a **macOS test harness**, where MLX has a real Metal GPU.
That is how the loaded-model and inference-session seams are verified without a
physical iPhone.

## Running the tests

Use `xcodebuild`, not `swift test`. Xcode's build system compiles the MLX Metal
shader library (`default.metallib`); plain `swift test` does not, so MLX fails
to initialize there.

```sh
xcodebuild test -scheme HibikiCore -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation
```

MLX ops in tests run on the **CPU** backend (see `MLXTestSupport`): parity is a
numerical property, and the Metal GPU path is exercised on the device (ticket
#26).

Tests that load real weights read the bundle from `$HIBIKI_ARTIFACTS`, else
`artifacts/hibiki-1b-mlx-bf16/` at the repo root. That bundle is ~4 GB and
gitignored; download it first (see `hibiki_mlx/hibiki_mlx/download.py`), or those
tests skip.

## Layout

- `Sources/HibikiCore/` — config decode, safetensors indexing, artifact-bundle
  validation; MLX modules are added here as the native port proceeds.
- `Tests/HibikiCoreTests/` — the macOS seam tests.
