# Hibiki 1B MLX inference: core implementation reference

This document is the implementation context for an on-device inference library around `kyutai/hibiki-1b-mlx-bf16`. It records the released artifact contract, the streaming state machine, tensor shapes, and MLX-specific runtime constraints. It also calls out places where Kyutai's paper and public reference implementations do not agree. Treat those items as parity-test requirements, not as details to guess. Confirmed public behavior and package constraints are summarized in [the Python package contract](./python-package.md).

> **Project direction:** this repository reimplements inference directly with MLX. `moshi_mlx` and `moshi-swift` are read-only behavioral references, not runtime dependencies and not code to wrap, import, or copy blindly. A later iOS target will be a separate native MLX Swift implementation sharing the same artifact and inference contracts.

Hibiki is a decoder-only, multistream model that consumes French speech and emits English speech plus aligned English text. The released 1B model is the paper's 1.7B-parameter **Hibiki-M**: eight Mimi RVQ codebooks for source speech, eight generated codebooks for target speech, and one text stream, all clocked at 12.5 Hz. The model card reports a 1.1 kbps audio rate and training sequences up to 120 seconds; the model is French-to-English only. ([model card](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/README.md#L17-L36), [training limit](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/README.md#L51-L64))

## Source baselines

All implementation links below are immutable revisions.

| Source | Revision used | Role |
| --- | --- | --- |
| Hibiki repository | [`f1cf929`](https://github.com/kyutai-labs/hibiki/tree/f1cf9293e35c1dceffbe60dd325bdd702bc8305e) | Product-level description and Hibiki-specific Rust reference. The repository says that the actual Python/MLX implementation lives in Moshi. ([README](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/README.md#L57-L61)) |
| Moshi checkpoint release | [`dd6b9ff`](https://github.com/kyutai-labs/moshi/tree/dd6b9fffd613e5a2c64166a7ec09b121be09877b) (`moshi_mlx` 0.2.1) | Checkpoint-era MLX implementation. Hibiki explicitly requires `moshi_mlx >=0.2.1`; that release required MLX `>=0.22,<0.23`. ([Hibiki README](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/README.md#L83-L102), [package metadata](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/pyproject.toml#L1-L18)) |
| Moshi published baseline | [`f257343`](https://github.com/kyutai-labs/moshi/tree/f2573439ed70a8384b3f390e561f9c325fc890c7) (`moshi_mlx` 0.3.0) | Latest published package verified on 2026-08-17. It added MLX batching and requires MLX `>=0.26,<0.27`. ([version](https://github.com/kyutai-labs/moshi/blob/f2573439ed70a8384b3f390e561f9c325fc890c7/moshi_mlx/moshi_mlx/__init__.py#L6-L12), [dependencies](https://github.com/kyutai-labs/moshi/blob/f2573439ed70a8384b3f390e561f9c325fc890c7/moshi_mlx/pyproject.toml#L1-L18), [PyPI](https://pypi.org/project/moshi-mlx/0.3.0/)) |
| Moshi upstream snapshot | [`e6a55d2`](https://github.com/kyutai-labs/moshi/tree/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362) | Later `main` snapshot used below to inspect current API and cache behavior. Its version string remains 0.3.0, so do not mistake it for the exact published source archive. |
| Moshi Swift historical snapshot | [`df64ffd`](https://github.com/kyutai-labs/moshi-swift/tree/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a) | Experimental MLX Swift implementation and iOS proof of concept, retained as a historical behavior and platform reference for a possible later native Swift implementation. ([README](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/README.md#L3-L16)) |
| Hugging Face model | [`b3d6291`](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/tree/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae) | The exact config, LM weights, Mimi weights, and SentencePiece model targeted here. |
| Hibiki paper | [arXiv `2502.03382v2`](https://arxiv.org/html/2502.03382v2) | Architectural intent, training/inference protocol, evaluation, and on-device result. |
| MLX | [`9a79573`](https://github.com/ml-explore/mlx/tree/9a795735ad9a42664e08f42361b405ed570bcf1a) | Official MLX semantics and APIs referenced in the runtime guidance. |

Do not copy a moving `main` implementation without recording the revision. The checkpoint's tensor contract is stable, but helper APIs have already changed: release 0.2.1 hardcodes generation batch size one, while current `LmGen` accepts `batch_size`. ([0.2.1 generator](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/models/generate.py#L14-L40), [current generator](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L14-L46))

### Moshi Swift: historical iOS reference

`moshi-swift` is **read-only study material**, not a package or runtime dependency, and not source to copy blindly. The README claims Moshi and Hibiki variant support; its value here is showing one historical native Apple implementation: both causal Mimi and the multistream LM run through MLX Swift, with 24 kHz PCM processed in 1,920-sample frames, the delayed 17-stream schedule advanced one frame at a time, and completed target tokens decoded back to PCM. ([README](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/README.md#L10-L16), [Mimi configuration and streaming API](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/Mimi.swift#L22-L49), [1,920-sample loop](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiCLI/RunMoshi.swift#L111-L127), [app frame loop](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L653-L689), [scheduler](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/LM.swift#L358-L451))

For a later native iOS/Swift implementation, study its ownership seams: Mimi holds encoder/decoder streaming state, the LM holds Temporal and Depth caches, and `LMGen` owns the delayed token schedule. Session reset clears Mimi, the schedule, and Temporal caches; the Depth cache is scratch state reset at the start of every frame. ([Mimi reset](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/Mimi.swift#L121-L133), [Depth reset](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/LM.swift#L30-L66), [session reset](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L648-L651)) The snapshot is explicitly experimental, pins MLX Swift 0.21.2, and loads separately repackaged/quantized artifacts with hardcoded weight conversion. Although its project targets iOS 18.1, macOS 15, and visionOS 2.1, its microphone source warns that it is probably macOS-specific and unlikely to work on iOS. ([platform targets](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/moshi.xcodeproj/project.pbxproj#L1009-L1035), [audio warning](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/AudioRT.swift#L26-L27), [dependency pin](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/moshi.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved#L14-L20), [artifact loading](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L127-L180), [Mimi weight conversion](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L211-L272))

This snapshot also has no dedicated Hibiki model type, configuration, or checkpoint: the Hibiki UI path constructs `MoshiModel` with `moshi1b` and a `moshi-*` artifact. Treat the two-stream scheduler and Swift/MLX structure as historical evidence, not proof of compatibility with this project's BF16 bundle or authoritative Hibiki CFG/end-of-stream behavior. ([UI model selection](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L23-L59), [model construction](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L363-L374), [Hibiki-labelled route](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L619-L645))

## System contract at a glance

```text
24 kHz mono float PCM
       │  1,920 samples = 80 ms
       ▼
causal Mimi encoder ──► 8 source RVQ ids
       │                    │
       │                    ▼
       │          delayed multistream scheduler
       │          [text + 8 target + 8 source]
       │                    │
       │                    ▼
       │          16-layer Temporal Transformer
       │             ├──► one text token
       │             └──► 6-layer Depth Transformer × 8 slices
       │                         │
       │                         ▼
       └──────────────── 8 target RVQ ids
                                 │
                                 ▼
                         causal Mimi decoder
                                 │
                                 ▼
                       1,920 samples of PCM
```

One complete frame must be processed every 80 ms to sustain real time. That budget includes source encoding, the temporal step, eight sequential depth slices, target decoding, MLX evaluation/synchronization, and application scheduling. The constant 12.5 Hz clock is part of the model, not a configurable chunk size. ([paper codec model](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS1), [MLX loop](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L89-L133))

## Released model contract

### Artifact set and integrity

Load all four files from one pinned Hugging Face revision. Mixing a config from one revision with weights from another must be an error.

| Artifact | Purpose | Released size and LFS SHA-256 |
| --- | --- | --- |
| [`config.json`](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json) | Architecture, stream layout, delays, and conditioner | Small JSON file |
| [`hibiki-mlx-dc2cf5a5@80.safetensors`](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/hibiki-mlx-dc2cf5a5%4080.safetensors) | BF16 LM and Depth Transformer weights | 3,600,043,224 bytes; `2d1baa58b2003aef24a034cdec5bc8c6b4c6d14d0d50e530c42708e62e0b30d9` |
| [`mimi-dbaa9758@125.safetensors`](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/mimi-dbaa9758%40125.safetensors) | Causal Mimi codec weights | 384,644,900 bytes; `31c14cf365353131094e8248150c6fe58e8642cf91899c50d9e450f861630e55` |
| [`tokenizer_spm_48k_multi6_2.model`](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/tokenizer_spm_48k_multi6_2.model) | 48k SentencePiece text tokenizer | 857,314 bytes; `c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697` |

The released files alone occupy about 3.99 GB (3.71 GiB) on disk before package code or caches. The model weights are CC-BY 4.0; preserve attribution obligations in any redistribution. ([Hibiki licensing](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/README.md#L129-L149))

### Architecture values

The artifact config is authoritative for construction. The paper's Hibiki-M description agrees on a 2,048-dimensional, 16-layer temporal backbone and eight codebooks per stream. ([config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L5-L33), [paper architectural hyperparameters](https://arxiv.org/html/2502.03382v2#S4.SS1))

| Component | Value | Implementation meaning |
| --- | --- | --- |
| Temporal Transformer | `d_model=2048`, 16 layers, 16 heads, causal | One time-axis step per 80 ms frame; head dimension is 128. |
| Temporal attention | RoPE base 100,000; `context=500` | Attend over at most 500 prior frames, i.e. 40 seconds. The 40-second statement is also explicit in Hibiki's README. ([source](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/README.md#L48-L55)) |
| Temporal FFN | gated SiLU, config `hidden_scale=4.125` | Moshi's parser constructs the gated family with an effective branch width of 5,632 (`11*d_model/4`), a 2×5,632 input projection, and a 5,632→2,048 output projection. Do not build a plain 4.125× GELU MLP. ([parser](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L52-L76), [gated MLP](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/transformer.py#L160-L175)) |
| Text | 48,000 output tokens, 48,001 input embeddings | The extra input-only id is the start token. |
| Audio | `n_q=16`, cardinality 2,048 | Sixteen temporal audio streams are **8 target + 8 source**, not 16 generated RVQs. |
| Depth Transformer | 1,024 dimensions, 16 heads, 6 layers, configured FFN 4,224, context 16, no positional embedding | Gated branch width is 2,816 (`2*4224/3`). It samples eight target codebooks sequentially within each temporal frame. Its cache is reset at the start of every frame. ([depth sampler](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L236-L285), [gated MLP](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/transformer.py#L160-L175)) |
| Conditioning | `description` LUT, 16-dimensional input embedding projected to model width | Valid labels are `very_bad`, `bad`, `neutral`, `good`, and `very_good`; the projected condition is added at every time step. ([config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L34-L47), [conditioner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/conditioner.py#L125-L176)) |

The paper describes the training-time Depth Transformer as operating over both target and source halves. At inference, source predictions are skipped and the actual encoded source ids are teacher-forced. The released MLX config therefore has 16 temporal audio streams but only eight depth slices. ([paper multistream model](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS3), [paper architectural detail](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS4), [MLX config conversion](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L101-L138))

## Tensor and token contracts

Use batch-time-channel notation explicitly at every boundary. Do not let transposes leak across the codec/model interface.

| Boundary | Required shape and dtype for one streaming frame | Evidence |
| --- | --- | --- |
| Input PCM | NumPy `float32 [B, 1, 1920]`, mono at 24 kHz | The reference reads/resamples to 24 kHz and slices 1,920 samples. ([MLX runner](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L89-L94), [loop](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L105-L121)) |
| Mimi encode output | Raw codec convention `[B, 8, T]`; for a steady-state frame `T=1` | The Python binding returns a rank-3 code tensor and the runner transposes it before the LM. ([binding](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L134-L179), [transpose](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L141-L147)) |
| Generator source input | MLX `int32 [B, 8]` | Eight source codebooks are stored in the non-generated half of the sequence. ([generator](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L62-L86)) |
| Persistent token schedule | MLX `int32 [B, 17, max_steps]` in the reference | Axis 1 is one text stream, eight generated target streams, then eight source streams. A production implementation may use a bounded ring buffer, provided delay semantics are identical. ([allocation](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L27-L43)) |
| Temporal input | Text ids `[B,1]`; logically 16 audio id columns `[B,1]`; summed embedding `[B,1,2048]` | Current upstream internally represents each audio column as `[1,B]` and transposes its embedding. Normalize this oddity at the library boundary. The model sums text, audio, and condition embeddings before the temporal transformer. ([scheduler representation](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L81-L105), [sampling path](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L457-L488)) |
| Text logits/output | Logits `[B,1,48000]`; sampled id `[B,1]` in current code | Text is sampled before audio and is the first Depth Transformer input. ([sampling path](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L478-L498)) |
| Depth output | `int32 [B,8,1]` | One target token for each generated codebook, generated autoregressively across depth. ([depth output](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L255-L285), [shape check](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L95-L114)) |
| Mimi decode input/output | NumPy `uint32 [B,8,1]` to PCM `[B,1,1920]` in steady state | The runner adds the singleton time axis and the real-time client requires 1,920 output samples. ([offline runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L149-L158), [real-time output](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/local.py#L201-L234)) |

### Special ids

| Domain | Id | Meaning |
| --- | ---: | --- |
| Text | `48000` | Input-only start/BOS id; used as previous text on the first temporal step. ([source](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L68-L77)) |
| Text | `3` | Padding/no-text token; do not detokenize or emit it. It is also named in the artifact config. ([config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L5-L10), [Hibiki Rust state](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L103-L124)) |
| Text | `0` | EOP/control token in the Rust generator; the MLX runners also suppress it. ([Rust config](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L103-L114), [MLX output filter](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L147-L154)) |
| Audio | `0..2047` | Depth output / Mimi code range. The Depth head has 2,048 outputs. ([model construction](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L236-L250)) |
| Audio | `2048` | LM-only padding id; it is embedded but cannot be emitted by the Depth head. ([properties](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L140-L147)) |
| Scheduler | `-1` / `-2` | MLX-internal zero/no-input and ungenerated sentinels. Never pass them to Mimi. ([generator](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L48-L60)) |

Do not decode text by blindly calling `id_to_piece()` and replacing `▁` if correctness matters. The Rust Hibiki reference incrementally decodes the previous/current id pair and emits the suffix, which handles SentencePiece composition more safely; accumulating ids and diffing the decoded text is another valid approach. ([incremental text decoder](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L23-L44))

## Streaming schedule

The config delay vector is:

```text
text:   [0]
target: [0, 2, 2, 2, 2, 2, 2, 2]
source: [0, 2, 2, 2, 2, 2, 2, 2]
```

The first codebook of each audio stream is the semantic level. Codebooks 2–8 are acoustic levels delayed by two 80 ms frames. This is the paper's acoustic-delay transform and is encoded exactly in the released config. ([paper](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS2), [config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L27-L33))

For LM step `t`, the reference scheduler does the following:

1. Store current source ids in streams 9–16 at schedule position `t`.
2. Feed BOS text at `t=0`; otherwise feed generated text `W[t-1]`.
3. For each audio stream `q`, feed schedule position `t-1-delay[q]`, or audio PAD when the index is negative.
4. Sum all embeddings and the condition; run one Temporal Transformer step.
5. Sample `W[t]` from the 48k text head.
6. Reset the Depth Transformer scratch cache, then sample target RVQ levels 1–8 sequentially, beginning with `W[t]` and then the preceding generated RVQ id.
7. Store target semantic id at position `t`; store each target acoustic id at position `t-2`.
8. Return the newest target frame for which all eight codebooks are complete: position `t-2`. No audio frame is available for the first two LM steps.

These index operations are implemented directly by `LmGen`; the paper gives the corresponding time/depth factorization. ([scheduler](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L62-L116), [completed-frame lookup](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L138-L148), [paper](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS4))

The text and audio returned by one API call are therefore on different timeline positions: the newly sampled text belongs to frame `t`, while the newly decodable audio belongs to frame `t-2`. A `StepResult` should carry `text_frame_index=t` and `audio_frame_index=t-2` instead of pretending the outputs are synchronous. The two-frame scheduler delay is 160 ms; it is separate from the learned linguistic translation lag measured by LAAL.

Text timestamps lie on the 80 ms grid. Define whether a word timestamp is the first piece, last piece, or decoded-boundary time and keep that policy outside the model core. The model card's “timestamped text” follows from text and audio sharing the constant frame clock. ([model card](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/README.md#L28-L36))

## State and cache ownership

A live inference session owns all of the following mutable state:

- frame index and previous text id;
- delayed token schedule or an equivalent bounded ring buffer;
- all 16 Temporal Transformer layer KV caches and absolute RoPE offset;
- the Depth Transformer scratch cache, reset once per temporal frame;
- Mimi encoder and decoder streaming convolution/Transformer state;
- sampling RNG state;
- any PCM ingress/egress ring buffers and end-of-stream state.

Kyutai's MLX `Lm` allocates the Temporal and Depth caches on the model object itself. Consequently, a model instance is mutable session state and is not re-entrant: two independent streams cannot safely interleave calls through one `Lm` unless cache state is separated or batched deliberately. ([cache construction](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L288-L323))

The Temporal Transformer uses a rotating cache with a hardcoded capacity of 4,096 frames, while attention trims computation to the config's 500-frame local context. At 12.5 Hz those are 327.68 seconds of cache address space and 40 seconds of attended history, respectively. Current source contains a TODO that its attention trimming is incorrect for `RotatingKVCache`; long-session behavior near/after wraparound requires a parity test. ([config conversion](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L52-L76), [attention trim and TODO](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/transformer.py#L136-L155), [rotating cache construction](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/transformer.py#L272-L290))

`reset_session()` must reset/recreate the generator schedule, every Temporal cache, the Depth scratch cache, both Mimi directions, and per-session RNG. Resetting only `LmGen.step_idx` is incorrect. Mimi exposes `reset()` in the Python binding and resets encoder, decoder, Transformer, upsample, and downsample state internally. ([Python binding](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L200-L235), [Mimi state reset](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/mimi.rs#L214-L231))

## Codec references and target

Mimi is causal and streaming, with 24 kHz mono input, a 12.5 Hz output frame rate, 2,048-entry RVQ codebooks, and eight codebooks selected for Hibiki-M. The first codebook carries semantic information and later codebooks progressively add acoustic detail. ([paper](https://arxiv.org/html/2502.03382v2#S3.SS1.SSS1), [Mimi configuration](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L43-L100))

The reference `moshi_mlx.run_inference` does **not** run Mimi in MLX. `rustymimi.Tokenizer` fixes Candle's device to CPU and defaults to `f32`; its streaming wrapper runs encoder and decoder on worker threads. The practical reference pipeline is therefore CPU Mimi plus MLX model (normally GPU), with NumPy/MLX conversion at every frame. ([Tokenizer device](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L103-L131), [worker threads](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L238-L315))

This project's Python target instead implements Mimi locally with MLX and loads the released weights directly. The CPU `rustymimi` path above is evidence for streaming behavior and parity tests, not the intended runtime architecture.

Both `encode_step()` and `decode_step()` can return no output while causal state is filling. Advance the LM only when an encoded source-code frame exists, and enqueue PCM only when the decoder returns it; do not convert `None` into a token/PCM frame. ([encode binding](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L154-L179), [decode binding](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L200-L230))

Keep the codec behind a narrow stateful interface such as `encode_frame(float32[B,1,1920]) -> uint32[B,8,1] | None`, `decode_frame(uint32[B,8,1]) -> float32[B,1,1920] | None`, and `reset()`. An all-MLX Mimi exists upstream, but it has its own cache and weight-conversion path; changing backends requires frame-by-frame parity and state-reset tests. ([MLX Mimi](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/mimi.py#L90-L178), [weight conversion](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/mimi.py#L188-L253))

For real-time I/O, do not encode, run the LM, or decode in an audio callback. Kyutai's local client uses fixed 1,920-sample sound-device blocks, queues, separate processes, and codec worker threads. Preserve bounded queues and backpressure metrics; an ever-growing output queue can hide loss of real-time performance. ([real-time client](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/local.py#L147-L239))

## Loading and initialization

Recommended load sequence:

1. Resolve a pinned Hugging Face revision and verify the four artifact names and, when distributed by this project, their SHA-256 values.
2. Parse and validate config invariants before allocating the model: `n_q=16`, `dep_q=8`, 17 delays including text, 500-frame temporal context, 2,048 audio cardinality, and 48,000 text cardinality.
3. Construct the local MLX Mimi, Temporal Transformer, and Depth Transformer modules from the validated model spec.
4. Cast floating parameters to `mx.bfloat16` before weight evaluation.
5. Load the safetensors with `strict=True`; MLX then verifies exact parameter names and shapes.
6. Load SentencePiece and strict-load the local Mimi module.
7. Construct both `very_good` and, if real CFG is enabled, `very_bad` condition embeddings.
8. Warm up fixed-shape model and codec paths, then reset all session state unless the warmup silence is an intentional prefix.

The checkpoint-era MLX runner follows the same core order: config, model construction, BF16 cast, optional checkpoint-format quantization, strict load, tokenizer/codec, condition, then warmup. ([runner](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L51-L103))

MLX is lazy: constructing a module records initializers but does not evaluate them. Loading lower-precision weights before evaluation avoids a full float32 initialization peak. `Module.load_weights(strict=True)` checks extra/missing names and every shape. ([MLX lazy loading guidance](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/lazy_evaluation.rst#L48-L63), [strict loading implementation](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/python/mlx/nn/layers/base.py#L123-L207), [dtype cast](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/python/mlx/nn/layers/base.py#L613-L632))

Warmup is part of the runtime contract. `Lm.warmup()` executes one sample and resets the Temporal caches; the real-time frontend pushes several silent codec/model frames without resetting afterward, which also creates a silence prefix. Decide explicitly whether warmup is compile/allocation warmup followed by a full reset or deliberate pre-roll. ([LM warmup](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L530-L544), [full real-time warmup](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/local_web.py#L79-L99))

## Sampling and classifier-free guidance

Expose text and audio sampling separately. They have different vocabularies and validated settings.

| Profile | Text | Audio | CFG |
| --- | --- | --- | --- |
| Public MLX/Rust runner | temperature 0.8, top-k 25 | temperature 0.8, top-k 250 | default coefficient 1.0 (disabled) |
| Paper Audio-NTREX setting | temperature 0.8, top-k 50 | temperature 0.8, top-k 250 | `gamma=3.0` |
| Paper CVSS setting | temperature 0.1, top-k 50 | temperature 0.8, top-k 250 | `gamma=3.0` |

The public-runner profile is hardcoded in both the MLX and Hibiki Rust examples. The paper reports its settings as cross-validated and warns that low text temperature can emit EOS prematurely. Do not present one profile as universal; make both samplers configurable and record the chosen profile in diagnostics. ([MLX runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L128-L135), [Rust runner](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L78-L85), [paper inference configuration](https://arxiv.org/html/2502.03382v2#S4.SS5))

Current Moshi's sampler uses compiled MLX categorical/top-k/top-p/min-p functions and captures MLX random state as input/output. `temperature=0` is greedy; otherwise only one of top-k, top-p, or min-p is selected by priority. ([sampler](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/utils/sampling.py#L10-L16), [dispatch](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/utils/sampling.py#L130-L165))

### Intended CFG

The paper defines voice-transfer CFG as:

```text
guided_logits = gamma * logits(very_good) + (1 - gamma) * logits(very_bad)
              = gamma * logits_good - (gamma - 1) * logits_bad
```

It evaluates both conditions as a batch of two. `gamma=1` is the positive conditioned model; larger values strengthen voice similarity, but the paper shows excessive guidance can degrade intelligibility and translation. The Hibiki Rust app constructs `very_good` and `very_bad` rows explicitly and its generation state combines their text and depth logits. ([paper CFG](https://arxiv.org/html/2502.03382v2#S3.SS3), [paper ablation](https://arxiv.org/html/2502.03382v2#S4.SS6), [Rust condition batch](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L88-L123), [Rust logit combination](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/lm_generate_multistream.rs#L225-L270))

### MLX CFG discrepancy

Do **not** assume `moshi_mlx.run_inference --cfg-coef 3` implements the formula above. Both the checkpoint-era and current MLX runner create only `condition_tensor("description", "very_good")`. `Lm._sample()` adds that condition and then tiles the already-conditioned input when `cfg_coef != 1`; the two rows therefore appear identical, making guidance a no-op apart from numerical noise. ([runner condition](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L113-L125), [tiling and combination](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L469-L498))

A correct implementation must form distinct positive/negative conditioned rows before the Temporal Transformer, preserve the two rows through all eight depth slices, combine both text and audio logits, and collapse back to logical batch `B`. Add tests that `gamma=1` matches the positive branch and that positive/negative logits are measurably different before enabling CFG in the public API.

## End of input and draining

The paper's intended protocol is explicit: after source speech ends, send an EOS token to the input, continue sampling, and stop when the model produces its own EOS. ([paper inference configuration](https://arxiv.org/html/2502.03382v2#S4.SS5))

The public examples do not fully implement that protocol:

- MLX processes exactly `floor(num_samples / 1920)` frames, with no partial-frame padding, input EOS, post-input sampling loop, or two-frame acoustic-delay drain. ([MLX loop](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L128-L158))
- The Hibiki Rust example appends 12,000 zero samples (0.5 seconds) and then processes those frames, but it likewise does not explicitly inject EOS or sample until output EOS. ([Rust input padding](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L51-L64), [Rust loop](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L127-L168))
- `LmConfig.audio_eos_token` aliases id 2047, which is also in the 2,048-way Mimi/depth output range; the examples do not demonstrate a reliable audio-EOS injection path. ([properties](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L140-L147), [depth output size](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L236-L250))

The checkpoint-specific learned EOS contract remains unresolved, so version one deliberately uses a different, named policy. `finish()` zero-pads one non-empty partial source frame, advances exactly six further silent frames, returns only target rows complete across all eight codebooks, and discards and counts the newest two incomplete delayed rows. A repeated call returns no duplicate output and reports `already_finished`. This is a deterministic package contract, not paper-equivalent EOS behavior and not an exact copy of any one public runner; the reference comparison is recorded in [the finalization research note](./research/finalization-reference-behavior.md).

## MLX runtime implications

### Evaluation boundaries

MLX operations are lazy until `mx.eval`, conversion to NumPy, scalar `.item()`, printing, or array-dependent Python control flow. A streaming step needs a deliberate evaluation boundary so the graph neither grows across frames nor synchronizes repeatedly within a frame. ([official lazy-evaluation guide](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/lazy_evaluation.rst#L8-L14), [implicit evaluations](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/lazy_evaluation.rst#L64-L91), [scalar/control-flow warning](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/lazy_evaluation.rst#L110-L144))

The reference generator performs `.any()` checks on MLX arrays inside Python control flow for each delayed stream and later calls `.item()` / converts audio to NumPy. Those are useful debug assertions but can introduce hot-loop synchronization. In a production path, keep shape/index invariants as Python checks, gate data-dependent token assertions behind a debug mode, and evaluate all outputs and updated cache state together once per frame before crossing to SentencePiece/NumPy. ([generator checks](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L81-L105), [runner conversions](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L147-L158))

### Compilation

`mx.compile` can fuse work and reduce graph/runtime memory, but the first call compiles and changes in shape/dtype can retrace. Hold frame and batch shapes constant, compile outside the loop, and warm every supported logical batch/CFG shape before real-time I/O. ([official compile guide](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/compile.rst#L8-L14), [cache/recompile behavior](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/compile.rst#L42-L77))

Compiled functions are intended to be pure. Transformer KV cache and RNG mutations must be explicit captured inputs/outputs or explicit return values; otherwise compilation can capture stale state. Upstream sampling correctly declares MLX random state as both input and output. ([compile state semantics](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/compile.rst#L173-L236), [sampling capture](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/utils/sampling.py#L10-L16))

### KV-cache allocation and memory

MLX recommends growing autoregressive caches in fixed chunks and updating them in place; naive concatenation becomes quadratic in copied elements. Moshi's cache uses 256-position growth chunks, matching the official recommendation. ([MLX KV-cache guide](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/kv_cache.rst#L3-L36), [Moshi cache](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/kv_cache.py#L12-L49))

For BF16 Temporal KV, the theoretical storage per logical batch item per frame is:

```text
16 layers × 2 (K,V) × 16 heads × 128 head_dim × 2 bytes = 131,072 bytes/frame
```

That is about 64 MiB for 500 frames and 512 MiB at the reference cache's 4,096-frame capacity. True CFG uses physical batch `2B`, so it doubles those numbers. Version one instead allocates a 512-position rotating cache, attends to the latest 500 positions, and preserves absolute RoPE offsets across wraps. This choice must pass the 120-second, 1,500-frame parity and bounded-memory test. The dimensions and cache layout follow directly from the model config and `[B, heads, time, head_dim]` cache allocation. ([cache layout](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/kv_cache.py#L88-L147), [head dimension](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/modules/transformer.py#L13-L40))

Apple silicon uses unified memory for MLX arrays, so model weights, caches, and working buffers compete with the application and OS in one pool. Unified memory removes explicit MLX CPU/GPU copies, but it does not remove conversion/copy costs between Candle/NumPy and MLX in the reference CPU-codec pipeline. ([official unified-memory guide](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/unified_memory.rst#L3-L32))

### Quantization

The targeted Hugging Face artifact is BF16. The reference only changes module structure before load when the checkpoint filename explicitly ends in `.q4.safetensors` or `.q8.safetensors`; those paths expect correspondingly quantized weight files. Do not rename the BF16 file or assume that calling `nn.quantize` before loading it is a valid conversion. ([runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L88-L98))

MLX `nn.quantize` replaces eligible Linear and Embedding leaves in place. If this project later ships quantized artifacts, record mode, group size, bit width, excluded layers, converter revision, quality metrics, and the matching config/filename contract. ([official quantize API](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/python/mlx/nn/layers/quantized.py#L27-L100))

## Latency and performance targets

Distinguish four latency classes:

1. **Frame deadline:** warm end-to-end frame time must have p95 at or below 80 ms on the M1 Pro reference machine; report the maximum separately rather than failing on one isolated spike.
2. **Scheduler latency:** target acoustic levels add two frames (160 ms) before a complete output frame is decodable.
3. **Codec receptive-field/startup latency:** causal codec internals may require warmup before each input call produces a frame.
4. **Learned translation lag:** the model waits for linguistic context; this is seconds and is measured separately with End Offset/LAAL.

The paper reports Hibiki-M short-form End Offset 2.8 s / LAAL 3.5 s and long-form End Offset 2.3 s / LAAL 5.5 s in its evaluation table. Those are model-level task metrics, not compute latency. ([results table](https://arxiv.org/html/2502.03382v2#S4.T2))

Kyutai reports Hibiki-M faster than real time for a minute on an iPhone 16 Pro, including physical batch size two for CFG. It also says that training with sliding-window attention could improve real-time factor over time. This is evidence of feasibility, not a guarantee for a Python/MLX/Candle pipeline or other Apple hardware. ([on-device result](https://arxiv.org/html/2502.03382v2#S4.SS6.SSS1))

Benchmark cold and warm paths separately. At minimum record model/device/OS/MLX versions, CFG and logical batch, input duration, peak unified memory, Mimi encode p50/p95, temporal+depth p50/p95, decode p50/p95, end-to-end frame p50/p95/max, output-queue depth, real-time factor over time, and first-audio/first-text latency. Force an MLX evaluation at timing boundaries; otherwise lazy dispatch measures graph construction instead of execution. The official MLX compilation example likewise calls `mx.eval` inside its timing loop. ([MLX timing example](https://github.com/ml-explore/mlx/blob/9a795735ad9a42664e08f42361b405ed570bcf1a/docs/src/usage/compile.rst#L107-L137))

## Recommended library boundary

Keep mutable streaming policy out of the weight-bearing model module. A minimal separation is:

```text
ArtifactResolver
  └── pinned config/LM/Mimi/SentencePiece paths + integrity metadata

HibikiModel
  └── immutable architecture + loaded MLX parameters

HibikiSession
  ├── temporal/depth cache state
  ├── delayed-token scheduler
  ├── source encoder + target decoder state
  ├── samplers/RNG + positive/negative conditions
  └── push_pcm(), finish(), reset(), metrics()

StepResult
  ├── text token/piece + text_frame_index
  ├── PCM frame + audio_frame_index
  └── timing/backpressure diagnostics
```

`HibikiModel` and `HibikiSession` must be local implementations and must not wrap or import upstream runtime classes. Keep mutable caches, delayed history, and codec state inside the session so the public contract supports isolation, bounded memory, native Swift parity, and implementation-level testing.

## Required conformance tests

Before treating the core as production-ready, cover these invariants:

- reject config drift in cardinalities, codebook split, delay count/order, context, layer counts, and conditioner labels;
- assert every boundary shape/dtype in the tensor table, including batch `B>1` if supported;
- verify the exact delay schedule on synthetic token ids and confirm no output audio for the first two steps;
- verify `text_frame_index=t` and `audio_frame_index=t-2`;
- show chunked Mimi encode/decode parity with the selected reference runtime within an explicit tolerance;
- show reset determinism: a reset session with the same seed/input matches a fresh session;
- show independent sessions do not share KV, codec, or RNG state;
- test `gamma=1` parity and require distinct positive/negative logits before accepting `gamma>1`;
- compare the public-runner and paper sampling profiles on fixed fixtures;
- test non-multiple-of-1,920 PCM input and the fixed six-frame silence-tail `finish()` policy, including repeated finish;
- run exactly 120 seconds (the stated version-one limit), across the 500-frame context boundary and repeated wraps of the 512-position cache;
- benchmark warm p95 at or below the M1 Pro's 80 ms deadline, real-time factor at or below 1.0, and bounded queues/memory; report maximum frame time without using a single spike as the sole failure condition.

Parity integration fixtures should record artifact revision, implementation revision, seed, sampling profile, CFG behavior, PCM normalization, tail policy, and hashes of input/output tokens. Audio waveform equality across runtimes may be too strict; token equality and bounded PCM error should be separate assertions.

## Known gaps and deferred work

1. **CFG:** the paper and Hibiki Rust path use `very_good` versus `very_bad`; the public MLX path appears to duplicate `very_good`. True MLX CFG needs implementation plus parity tests.
2. **Learned EOS:** the paper specifies input/output EOS, while the public MLX and Rust examples use finite frame loops. Version one uses the fixed silence-tail policy above; the exact source EOS id and learned stopping behavior still require checkpoint-specific evidence.
3. **Audio EOS id:** current generic MLX config exposes 2047 as audio EOS, but 2047 is also in the emitted 2,048-way audio range. Do not rely on it without checkpoint-specific evidence.
4. **Long-session cache evidence:** version one supports exactly 120 seconds with 512 allocated positions and a 500-frame attention window. The implementation must still prove absolute-position correctness and bounded memory across repeated cache wraps.
5. **Codec parity:** the project target is a local all-MLX Mimi. It needs token, PCM, reset, and streaming-state parity tests against the pinned artifacts and historical references. A later native Swift implementation needs its own parity and performance measurements.
6. **Reference drift:** upstream Python and Swift revisions are pinned only for study. The Python package targets MLX 0.32 and the pinned BF16 bundle independently; a future Swift implementation must choose and verify its own native MLX and conversion versions.
7. **Quantization:** this artifact is BF16. Any lower-bit on-device variant needs an explicitly versioned conversion and quality/performance validation.

These gaps do not block a faithful frame-by-frame BF16 prototype with CFG disabled and an explicit silence-tail policy. They do block claiming paper-equivalent CFG, EOS completion, or unlimited live-stream behavior.
