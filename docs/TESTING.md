# Testing Hibiki inference engines

This document defines the shared verification contract for Hibiki inference engines. It applies to the first Python/MLX implementation and later native implementations such as MLX Swift. Runtime-specific test runners may differ, but they must use the same assets, preprocessing, observable behavior, and evidence requirements.

Tests focus on the model loader and inference session interfaces. Model internals and implementation language may change without requiring the shared tests to change.

For the expected inference flow, see [INFERENCE_ARCHITECT.md](./INFERENCE_ARCHITECT.md). Exact artifact, scheduling, parity, and performance contracts are recorded in [core-library.md](./core-library.md).

## What the test suites prove

| Suite | Purpose | Model weights required |
| --- | --- | --- |
| Fast | Validate artifact errors, PCM validation, session lifecycle, and pure result contracts. | No |
| Smoke | Prove that released artifacts can translate checked-in French audio into English text and PCM without runtime failure. | Yes |
| Conformance | Prove delayed scheduling, chunking invariance, reset determinism, and parity with pinned fixtures. | Yes |
| Long run | Prove the supported 120-second session and bounded cache/memory behavior. | Yes |
| Performance | Measure cold start, steady-state frame time, latency, and unified memory on named hardware. | Yes |

The smoke suite is a functional gate. It does not by itself prove translation quality, reference parity, or real-time performance.

## Test seams

Tests cross only these public seams:

1. **Model loading:** resolve and validate a pinned artifact bundle, load weights, warm the engine, or return an actionable error.
2. **Inference session:** push arbitrary PCM chunks, observe timestamped step results, finish, reset, and close.

Tests must not inspect private Transformer layers, mutate KV caches, seed scheduler internals, or mock Mimi. When a failure needs more localization, add diagnostic data to the public result or test a stable numerical module against an independently generated parity fixture.

## Model setup

Model-backed tests use:

```text
kyutai/hibiki-1b-mlx-bf16
```

Each test-runner adapter must receive a local artifact directory through explicit runtime configuration. For example, a command-line adapter may use `HIBIKI_MODEL_DIR`, while an application test target may use a launch argument or bundled test configuration. Automated tests must not silently download weights. If the artifact location is absent, local model-backed tests may skip with a clear reason; a required Apple-silicon CI or release lane must treat that skip as a failure.

Before starting a session, verify that the directory contains the matching configuration, Hibiki weights, Mimi weights, and SentencePiece model. Strict loading must reject missing, extra, mixed, or shape-incompatible artifacts.

Every implementation must map these stable case identifiers into its native test runner:

```text
smoke_fast
smoke_short
smoke_long
smoke_assets
conformance
long_run_120s
performance
```

The implementation may expose these as Pytest markers, XCTest plans, CLI cases, or another native mechanism. It may add narrower cases, but it must keep one quick model-backed smoke command and one command covering every source asset. Each implementation documents its concrete commands beside its test-runner adapter rather than changing this shared contract.

## Audio asset roles

### Primary smoke inputs

Only WAV files under a `source/` directory are French source inputs. Convert them to mono 24 kHz float32 PCM before passing them to the inference session.

| Input | Stored format | Approximate duration | Use |
| --- | --- | ---: | --- |
| `assets/short-form/source/cvss-fr2en-test-idx4562-19004869.wav` | 48 kHz, mono, int16 WAV | 10.008 s | Fast smoke and full asset smoke |
| `assets/short-form/source/cvss-fr2en-test-idx14345-20007437.wav` | 48 kHz, mono, int16 WAV | 10.248 s | Full asset smoke |
| `assets/short-form/source/cvss-fr2en-test-idx14410-20011543.wav` | 48 kHz, mono, int16 WAV | 10.176 s | Full asset smoke |
| `assets/short-form/source/cvss-fr2en-test-idx14603-20030929.wav` | 48 kHz, mono, int16 WAV | 10.176 s | Full asset smoke |
| `assets/short-form/source/cvss-fr2en-test-idx14695-20041791.wav` | 48 kHz, mono, int16 WAV | 10.272 s | Full asset smoke |
| `assets/long-form/source/5196_ea80c8e6-883d-4afe-841b-598ce7db3779.wav` | 24 kHz, mono, int16 WAV | 43.680 s | Long-form and full asset smoke |
| `assets/long-form/source/10887_ea80c8e6-883d-4afe-841b-598ce7db3779.wav` | 24 kHz, mono, int16 WAV | 45.740 s | Long-form and full asset smoke |
| `assets/long-form/source/9605_83f1360e-7775-4d36-89f6-60649041c935.wav` | 24 kHz, mono, int16 WAV | 49.160 s | Long-form and full asset smoke |
| `assets/long-form/source/6855_f3c3ea82-42ef-4c09-b4aa-544a4c95518b.wav` | 24 kHz, mono, int16 WAV | 57.120 s | Long-form and full asset smoke |
| `assets/long-form/source/3120_a63eabfc-d5aa-4353-84d0-9c5c068a1b38.wav` | 24 kHz, mono, int16 WAV | 59.320 s | Long-form and full asset smoke |

### Reference outputs

Each source file pairs with the 24 kHz mono Hibiki output under the neighboring `hibiki/` directory with the same basename. These are output references for listening comparisons and later quality metrics. They are not source inputs and are not waveform-exact goldens.

Generated speech is stochastic and may use different sampling, finalization, or historical runtime behavior. A smoke test must not require its samples to equal these reference WAV files.

## Deterministic input preparation

Every test must use the same preprocessing implementation:

1. Decode WAV samples without normalization, trimming, silence removal, or loudness processing.
2. Convert int16 to float32 using `sample / 32768.0`.
3. If a future fixture has multiple channels, downmix by averaging its channels. All current source fixtures are mono.
4. Resample to exactly 24,000 Hz with the project's pinned resampler and settings.
5. Preserve the full decoded duration.
6. Pass the resulting one-dimensional PCM to the session without writing a converted fixture back into `assets/`.

The resampler library, version, mode, and input/output sample counts belong in the smoke result manifest. Changing preprocessing requires deliberate fixture review because it can change every generated token.

## Required smoke runs

### Fast smoke

The default developer smoke uses the shortest short-form source fixture:

```text
assets/short-form/source/cvss-fr2en-test-idx4562-19004869.wav
```

Use a fixed seed, CFG coefficient `1.0`, the documented default text/audio sampling profile, frame-aligned 1,920-sample pushes, and the configured silence-tail finalization policy.

Expose this test as `smoke_fast` so it can be run after a model or session change without processing every asset.

### Full asset smoke

The release smoke runs all ten primary inputs listed above. Each input uses a newly created or fully reset session. Test order must not affect tokens or output.

Expose the five short inputs through a marker such as `smoke_short`, the five long inputs through `smoke_long`, and their union through `smoke_assets`. Keep them outside ordinary CPU-only test runs because loading the BF16 model and processing the complete asset set is expensive.

### Chunking smoke

Run the fast input twice with the same model, configuration, and seed:

- once in 1,920-sample chunks;
- once with a deterministic irregular chunk sequence containing both sub-frame and multi-frame chunks.

Both runs must produce identical text/audio token timelines and frame indices. Decoded PCM must meet the deterministic tolerance selected for the same MLX hardware/runtime. Transport chunking must not change inference behavior.

### Reset smoke

Run the fast input, reset the session, and run it again with the original seed. Compare both runs with a fresh session. Text tokens, target audio tokens, and frame indices must match exactly. PCM uses the documented deterministic numeric tolerance.

### 120-second long run

Construct the supported-duration input in memory from the 24 kHz long-form sources in their manifest order. Concatenate complete sources until the stream exceeds 2,880,000 samples, then truncate it to exactly 120 seconds. Do not finish or reset between source segments.

This fixture is a state, cache, and memory test rather than a translation-quality sample. It contains exactly 1,500 model frames and crosses the 500-frame attention window. Record the ordered source paths and truncation point in the result manifest.

## Automated smoke assertions

For every primary input, require all of the following:

- model loading and warmup complete before audio processing begins;
- no historical `moshi_mlx`, `rustymimi`, or `moshi-swift` runtime or compatibility layer is invoked;
- preprocessing produces finite, one-dimensional float32 PCM at 24 kHz;
- the session accepts the complete input and finishes without an unhandled exception;
- at least one non-special English text token and a non-empty decoded text string are produced;
- at least one non-empty mono float32 PCM block is produced;
- all output samples are finite and the complete output is not digital silence;
- text frame indices are strictly increasing;
- audio frame indices are strictly increasing whenever audio is present;
- complete audio is absent during the first two generation steps;
- when audio is present during normal generation, its frame index is the text frame index minus two;
- no codec no-output event is replaced with fabricated tokens or PCM;
- pushing after finish returns the documented lifecycle error;
- closing the session releases the model for the next asset;
- the test harness writes the result manifest and generated transcript/audio evidence successfully.

Do not make a smoke test pass by lowering assertions only for one asset. Investigate preprocessing, session state, finalization, or model compatibility first.

## Smoke evidence

Each model-backed run should retain:

- the decoded English transcript;
- the generated 24 kHz mono WAV;
- a machine-readable manifest;
- failure diagnostics when the run does not complete.

The manifest records at least:

- test case and pass/fail status;
- engine and model names;
- input path;
- seed, sampling, and finalization settings;
- transcript and generated-audio paths;
- input/output durations;
- processing time and real-time factor;
- error message when the run fails.

Generated evidence is test output, not source. Do not commit it unless it is intentionally reviewed and promoted into a versioned golden fixture.

## Expected metrics output

Each model-backed case writes a small JSON report. The same fields are used by Python, Swift, and later implementations.

```json
{
  "case_id": "smoke_fast",
  "status": "pass",
  "engine": "hibiki-mlx",
  "model": "kyutai/hibiki-1b-mlx-bf16",
  "input": {
    "path": "assets/short-form/source/cvss-fr2en-test-idx4562-19004869.wav",
    "duration_seconds": 10.008
  },
  "output": {
    "transcript_path": "<path-to-text>",
    "audio_path": "<path-to-wav>",
    "audio_duration_seconds": 9.92
  },
  "metrics": {
    "processing_seconds": 9.0,
    "real_time_factor": 0.90
  },
  "error": null
}
```

The numeric values above illustrate the shape and are not expected baselines. The runner fills them from the actual test.

A smoke case passes when:

- inference finishes without an error;
- generated text and audio are non-empty;
- audio samples are finite and not all silent;
- text/audio frame ordering passes the timestamp checks.

`processing_seconds` covers inference on prepared audio after model warmup. Real-time factor is `processing_seconds / input duration`. Record it for smoke runs, but use it as a gate only in the performance suite. A performance pass requires real-time factor at or below `1.0` and evaluated frame time below the 80 ms frame deadline on the declared reference device.

A suite writes one short summary containing the number of passed, failed, and skipped cases plus total processing time. Required cases must not be silently skipped.

## Conformance checks after smoke passes

Smoke success is followed by these engine-level checks:

- invalid artifact configurations fail before a session is created;
- identical input with the same seed is deterministic after reset;
- arbitrary PCM chunking is invariant;
- partial final frames are padded exactly once;
- text at step `t` and audio at step `t-2` retain distinct timestamps;
- the delayed schedule does not emit complete audio for its first two steps;
- a 120-second input crosses the 500-frame attention window without corrupt output or unbounded memory growth;
- fixed parity fixtures match expected source tokens, generated tokens, and decoded PCM tolerance;
- CFG values other than `1.0` fail until correct positive/negative conditioning is implemented.

Exact expected values must come from a pinned independent reference or an explicitly reviewed golden. Never generate an expected value with the code under test in the same test run.

## Manual review

For each short- and long-form pair, listen to the generated output beside the matching reference output and read the generated text. Record, but do not silently convert into automated thresholds:

- whether the output is intelligible English;
- whether it preserves the source meaning;
- obvious truncation or repeated speech;
- long unintended silence;
- clicks, discontinuities, or severe codec artifacts;
- whether finalization cuts off the last phrase.

Manual listening is required before promoting new golden fixtures. It does not replace deterministic token, state, and lifecycle tests.

## Performance verification

Run performance tests only after correctness and smoke tests pass. Force MLX evaluation at timing boundaries and report cold and warm paths separately.

The steady-state target is one complete encode, generate, and decode step within the 80 ms frame interval on the declared reference machine. Report p50, p95, and maximum time rather than only an average. Also report first-text latency, first-audio latency, real-time factor, peak unified memory, and whether frame time or memory grows across the input.

A slow smoke test is still a functional pass, but it is not a performance pass. Never hide missed deadlines by allowing an unbounded output queue.

## Failure triage order

When a smoke test fails, check in this order:

1. input decode, downmix, resampling, dtype, and sample count;
2. model artifacts and strict configuration/weight validation;
3. fresh-session and reset state;
4. Mimi source-token production;
5. text/audio frame indices and delayed scheduling;
6. generated target tokens before PCM decoding;
7. Mimi decoded PCM;
8. finalization state and remaining delayed positions;
9. MLX evaluation boundaries, timing, and memory.

Preserve the manifest and first failing frame. Do not diagnose translation quality from the final WAV alone.
