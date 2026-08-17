# Hibiki 1B MLX inference architecture

This document gives a high-level view of the inference flow. Exact model settings, tensor shapes, token ids, and source references live in [the core reference](./core-library.md).

This project is a clean MLX reimplementation. It loads the released Hibiki and Mimi weights directly and does **not** import, wrap, or depend on `moshi_mlx`. Upstream Moshi code is used only to understand expected behavior and build compatibility tests.

## Goal

The inference library receives streaming French speech and produces English speech plus aligned English text. The Mimi codec, Hibiki model, streaming scheduler, caches, and sampling are implemented locally with MLX.

## End-to-end flow

```text
French PCM audio
24 kHz mono
      │
      │ split into 1,920-sample / 80 ms frames
      ▼
Local MLX Mimi encoder
      │
      │ 8 French audio tokens
      ▼
Local MLX Hibiki generator
      ├──────────────► English text token
      │
      │ 8 English audio tokens
      ▼
Local MLX Mimi decoder
      │
      ▼
English PCM audio
24 kHz mono
```

The pipeline runs at 12.5 frames per second. All work for one frame should complete within 80 ms to remain real time.

## Main modules

| Module | Responsibility |
| --- | --- |
| Inference engine | Loads the released artifacts directly into local MLX modules. |
| Mimi codec | Local streaming MLX implementation that converts PCM to audio tokens and generated tokens back to PCM. |
| Hibiki generator | Local Temporal Transformer, Depth Transformer, sampling, KV cache, and delayed-stream implementation. |
| Inference session | Owns streaming state and coordinates framing, encoding, generation, text decoding, audio decoding, and finalization. |

The caller interacts with the engine and session only. Codec layouts, Transformer caches, and delayed-token scheduling remain internal.

## One inference step

For every 80 ms source frame:

1. The local MLX Mimi encoder converts French PCM into eight source tokens.
2. The local scheduler combines those tokens with the previous text and delayed audio history.
3. The Temporal Transformer produces the state for the current time step.
4. The model samples one English text token.
5. The Depth Transformer samples eight English audio tokens.
6. Once all delayed audio codebooks are ready, the local MLX Mimi decoder produces an English PCM frame.

The model processes 17 synchronized streams:

```text
1 text stream
8 generated English audio streams
8 supplied French audio streams
```

The first audio codebook has no delay; the remaining seven are delayed by two frames. Text produced at step `t` belongs to zero-based text frame `t`, while the complete audio returned during that call normally belongs to audio frame `t-2`. Model time is the frame index multiplied by 80 ms; it is distinct from the later wall-clock time at which delayed audio becomes available.

## Session state

One live session owns:

- incomplete input PCM;
- local Mimi encoder and decoder state;
- delayed source and target tokens;
- Temporal Transformer caches;
- previous and accumulated text tokens;
- sampling random state;
- frame counters and end-of-input state.

Model weights are immutable and reusable. Mutable Mimi state, Transformer caches, delayed tokens, and RNG state belong to the session rather than the model module. Version one may support one active session first, but the design must not make model-owned caches a permanent limitation.

## Session lifecycle

```text
Load and warm engine
        │
        ▼
Start session
        │
        ▼
Push PCM chunks ──► produce text and audio
        │
        ▼
Finish with the explicit silence tail
        │
        ▼
Reset or close
```

PCM chunks may have any length; the session buffers them into 1,920-sample frames internally. Model loading, downloads, and cold compilation happen before streaming begins. Version one pads a non-empty partial final frame, advances exactly six more silent frames, returns only complete delayed audio positions, and never describes this fallback as learned EOS completion.

## Runtime placement

The intended runtime is:

```text
CPU: PCM framing, text decoding, and application I/O
GPU through MLX: Mimi encode → Hibiki generation → Mimi decode
```

Inference should run on a dedicated worker. Microphone and playback callbacks should only move audio through bounded queues. Host conversions should happen only at the PCM and text interfaces.

## Reimplementation rule

The released `config.json`, safetensors, and SentencePiece model are the compatibility contract. Local modules should reproduce the required behavior directly:

- Mimi streaming encoder, quantizer, and decoder;
- Temporal and Depth Transformers;
- delayed source/target scheduling;
- KV-cache and streaming convolution state;
- text and audio sampling;
- condition handling and future classifier-free guidance.

Parity tests may compare outputs with upstream references, but production code must never call `moshi_mlx`.

A later iOS target should be a separate native MLX Swift implementation. It should reuse the same artifact, tensor, scheduling, and parity contracts while treating `moshi-swift` only as historical design evidence.

## Initial design constraints

- Target BF16 and logical batch size one first.
- Depend directly on MLX and the artifact formats, not `moshi_mlx`.
- Keep classifier-free guidance disabled until MLX correctly uses distinct `very_good` and `very_bad` conditions.
- Use an explicit silence tail to finish and drain output until learned EOS behavior is verified.
- Treat 120 seconds as the initially supported session length.
- Keep one active session per model instance.
- Keep the Temporal cache at 512 allocated positions while attending to the latest 500 and preserving absolute RoPE positions.
- Build the local all-MLX Mimi codec as part of the first frame loop. Add quantization and batching only after that BF16 path has parity and performance evidence.
