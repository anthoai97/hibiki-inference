# Frame indexing in the Python and Rust references

## Answer

Both references use **zero-based internal frame positions**. With a maximum audio delay of two frames, generation call `t` returns the text token for frame `t` and, once available, the completed audio row for frame `t - 2`.

| Generation call `t` | Text frame returned | Complete audio frame returned |
|---:|---:|---:|
| 0 | 0 | none |
| 1 | 1 | none |
| 2 | 2 | 0 |
| 3 | 3 | 1 |

Audio row `k` combines the undelayed codebook sampled at call `k` with the delayed codebooks sampled at call `k + 2`.

Neither public runner exposes these semantic frame numbers or timestamps. They print text and concatenate decoded audio into a WAV. Frame numbers and times would therefore be a new package API, not existing reference output.

## Python/MLX evidence

In both pinned Moshi revisions, `step_idx` starts at zero. A call samples and stores text at column `step_idx`, writes each generated audio codebook at `step_idx - delay`, and then increments `step_idx`. `last_audio_tokens()` reads `step_idx - 1 - max_delay`, so after call `t` it reads row `t - max_delay` ([checkpoint-era scheduler](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/models/generate.py#L30-L38), [checkpoint-era step and lookup](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/models/generate.py#L56-L112), [later scheduler initialization](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L34-L46), [later step and lookup](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L62-L148)). The scheduler reads the delay from model configuration rather than hardcoding `2`.

Both offline runners loop from zero internally but output only text pieces and concatenated PCM; the loop index never becomes public metadata ([`dd6b9ff` runner](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L105-L140), [`e6a55d2` runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L128-L166)).

## Rust evidence

The pinned Hibiki app configures `acoustic_delay: 2`, feeds one generation step at a time, prints the current text, and immediately asks for the latest completed audio row ([Hibiki runner](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L103-L165)). Its scheduler dependency is Moshi `0.5.2`; the scheduler citations below use an official Moshi snapshot whose workspace also declares version `0.5.2` ([Hibiki manifest](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/Cargo.toml#L12-L20), [Moshi workspace version](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/Cargo.toml#L1-L15)). This is a source-version match; Hibiki's crates.io dependency does not pin a Git commit.

That Rust scheduler initializes `step_idx` to zero, stores the sampled text at the current index, writes generated audio at `step_idx - delay`, and increments the index ([initialization](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/moshi-core/src/lm_generate_multistream.rs#L69-L123), [step](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/moshi-core/src/lm_generate_multistream.rs#L176-L277)). After the increment, `last_audio_tokens()` returns nothing for the first two calls and then reads `step_idx - acoustic_delay - 1`, which is row `t - 2` after call `t` ([completed-row lookup](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/moshi-core/src/lm_generate_multistream.rs#L319-L331)). The Hibiki runner exposes only generated text, aggregate timing, and the output WAV—not per-frame indices or timestamps ([runner output](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L169-L182)).

## Convention for Issue #1

Use zero-based `text_frame_index` and `audio_frame_index`. At normal generation call `t`:

- `text_frame_index = t`;
- no audio index exists for `t < 2`;
- otherwise `audio_frame_index = t - 2`.

If the package exposes model-timeline times, define them as frame **start** times: `time_seconds = frame_index * 0.08`. Thus the first text and audio frames both have model time `0.00 s`, even though audio frame 0 only becomes available during generation call 2. Keep processing or wall-clock latency in separate timing fields.
