# Python package contract

This document records the version-one package and session choices confirmed for Issue #1. Exact tensors, token identifiers, and source citations remain in [the core implementation reference](./core-library.md); canonical domain terms are in the root [context glossary](../CONTEXT.md).

## Package scope

- The project lives under `hibiki-mlx/`, installs as `hibiki-mlx`, and imports as `hibiki_mlx`.
- Use `uv` for project and dependency management.
- Support Python `>=3.13,<3.15`, Apple-silicon macOS 14 or newer, and `mlx>=0.32.0,<0.33`; lock MLX 0.32.0 for development, CI, fixtures, and performance reports.
- License the project code under Apache 2.0. Keep the downloaded model under CC-BY 4.0 and include its attribution in a package `NOTICE`.
- Do not bundle the roughly 4 GB artifact bundle in the Python distribution and do not publish the first version to PyPI.
- Include one offline command that accepts mono 24 kHz WAV input, prints translated text, and writes mono 24 kHz translated audio. It does not capture a microphone, run a server, provide a UI, or resample unsupported input.

## Artifact loading

- Load either an exact local bundle or a Hugging Face model ID with a revision. `huggingface_hub` is a normal dependency.
- Safe mode accepts the pinned BF16 revision `b3d6291f3dcf7954e1a502e4d66f32e3556f17ae` and verifies built-in hashes, names, configuration invariants, parameter names, and tensor shapes.
- An exact local bundle needs no extra manifest. Offline mode must not attempt a network request.
- `allow_unsafe=True` relaxes only the known-revision and known-hash requirement. It never relaxes file, configuration, name, or shape validation.
- Loading, allocation, compilation, and warmup complete before a session starts. Warmup is followed by a full state reset and does not become a prefix of user audio.

## Session input and results

- A session declares the only supported sample rate, 24,000 Hz, when it is created.
- `push_pcm()` accepts one-dimensional, normalized NumPy `float32` mono PCM in arbitrary chunk sizes and always returns a list, which may be empty.
- Invalid shape, dtype, sample rate, non-finite value, range, or configuration rejects the entire call without advancing session state. A call that would cross 120 seconds is rejected the same way. An empty push returns an empty list.
- Public errors distinguish artifact-bundle failures, incompatible configuration, invalid audio input, invalid session state, and an already-busy loaded model.
- A generation result contains the sampled text token, an append-only text delta, optional mono 24 kHz NumPy `float32` PCM, separate text and audio frame indices, their model times, and per-step timing.
- Frame indices are zero-based. At generation step `t`, the text frame is `t`; complete audio is absent for `t<2` and otherwise has audio frame `t-2`. Model time is `frame_index * 0.08` seconds and is separate from processing latency.
- `session.metrics()` exposes aggregate timing and memory diagnostics.

## Text and sampling

- Incremental text decoding is stateful. It buffers incomplete SentencePiece byte sequences and publishes only valid append-only text; it never rewrites earlier output or emits the replacement character `�`.
- Tokenizer control and chat-marker tokens remain available as raw diagnostic IDs but are hidden from user-facing text. An incomplete byte sequence at finish is discarded and reported.
- Text and audio have separate sampling configurations. Version one supports `temperature` and `top_k` only, defaults to temperature 0.8 with text top-k 25 and audio top-k 250, and treats temperature zero as greedy sampling.
- Invalid sampling values fail at session creation. A seed makes stochastic sampling reproducible, and reset restores the original random state.
- Classifier-free guidance values other than 1.0 are rejected until distinct positive and negative branches pass parity tests.

## Lifecycle and finalization

- A loaded model permits one active session. A second session attempt fails until the first session closes.
- Sessions support `push_pcm()`, `finish()`, `reset()`, `close()`, and Python context-manager cleanup.
- Reset is allowed before or after finish, restores the initial seed, and clears every mutable state. Reset after close fails. Close is safe to repeat and releases the loaded model.
- Push after finish fails without changing state.
- Finish pads one non-empty partial source frame with silence, advances exactly six additional 80 ms silent frames, and reports the complete final text, new results, completion reason, discarded incomplete text, and discarded incomplete audio positions.
- A repeated finish returns no duplicate text or audio and reports `already_finished`.

## Verification and completion

- Every implementation ticket includes its own quick tests. Ordinary quick tests require neither network access nor model weights.
- Isolated developer tooling may run pinned historical implementations to create small reviewed parity fixtures before local Mimi work begins. Fixtures contain only compact expected values and hashes, never model weights. The production package and normal test environment do not depend on those runtimes.
- Generated text and audio token fixtures match exactly. PCM comparison uses an explicitly measured numeric tolerance.
- Model-backed tests remain separate from the quick test command and use the existing checked-in French audio assets. They cover loading, end-to-end output, chunking invariance, reset, finalization, and the 120-second limit.
- On the M1 Pro 16 GB reference machine, warm streaming must achieve real-time factor at or below 1.0 and frame-time p95 at or below 80 ms. Record p50, p95, maximum, cold load, warmup, first output, and peak memory; a rare maximum spike is reported rather than used alone as a failure.
- A 120-second, 1,500-frame test must cross the 500-frame attention window without continued memory growth. The Temporal cache allocates 512 frame slots, attends to the latest 500, and preserves absolute positions.
