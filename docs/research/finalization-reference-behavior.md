# Finalization behavior in the pinned Hibiki references

## Question and scope

This note checks four proposed `finish()` rules against the primary reference code pinned by Issue #1:

1. pad a partial final PCM frame with silence;
2. append six silent 80 ms frames (0.48 seconds);
3. make repeated `finish()` calls harmless;
4. discard and report delayed audio positions that never become complete.

The inspected revisions are Moshi Python/MLX [`dd6b9ff`](https://github.com/kyutai-labs/moshi/tree/dd6b9fffd613e5a2c64166a7ec09b121be09877b) and [`e6a55d2`](https://github.com/kyutai-labs/moshi/tree/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362), Hibiki Rust [`f1cf929`](https://github.com/kyutai-labs/hibiki/tree/f1cf9293e35c1dceffbe60dd325bdd702bc8305e), and Moshi Swift [`df64ffd`](https://github.com/kyutai-labs/moshi-swift/tree/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a). The Rust app depends on `moshi = 0.5.2` ([manifest](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/Cargo.toml#L12-L20)); the scheduler citations below use an official Moshi source snapshot whose workspace declares version 0.5.2 ([workspace manifest](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/Cargo.toml#L10-L23)). The crates.io lock entry does not itself identify that Git commit, so this is a source-version mapping rather than a Git-pinned dependency claim.

## Short answer

The four proposed rules are **not** all reference behavior.

| Proposed rule | What the references do | Conclusion |
| --- | --- | --- |
| Pad a partial final frame | Python drops a remainder. Rust appends 12,000 zeros before framing, which indirectly completes any partial source frame at 24 kHz. Swift passes a short final slice into the streaming codec but never pads or flushes it. | Padding is a reasonable new package policy, but not a shared reference rule. |
| Add exactly six silent frames | Only Rust adds a tail: 12,000 zero samples, nominally 0.5 seconds at 24 kHz. Because framing uses floor division, this is not always exactly six additional frames. Python and Swift add no Hibiki tail. | Six frames is a new deterministic policy, not an exact copy of the reference. |
| Repeated `finish()` is safe | None of the references exposes `finish()` or defines repeated-finalization behavior. | Idempotency must be a package lifecycle decision. |
| Discard and count incomplete delayed positions | All runners emit only schedule rows complete across all target codebooks and then stop. The last two delay-dependent rows are not drained. None explicitly discards or counts them. | Not exposing incomplete rows matches observable reference behavior; explicitly counting them is a useful new diagnostic. |

## 1. Partial final PCM frame

### Python/MLX

The checkpoint-era runner computes `steps = num_samples // 1920`, iterates exactly that many full slices, and has no final partial-frame branch ([runner](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L89-L120)). Any remainder shorter than 1,920 samples is therefore dropped.

The later runner uses the same floor division and full-frame loop ([runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L128-L158)). It can add padding for configurations containing `stt_config`, but the pinned Hibiki BF16 configuration has no such field ([runner branch](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L102-L111), [complete pinned config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L1-L48)).

Neither the current MLX Mimi interface nor the Rust-backed Python binding provides an end-of-stream flush operation: they expose streaming encode/decode and reset operations only ([MLX Mimi methods](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/mimi.py#L129-L178), [Python binding methods](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/mimi-pyo3/src/lib.rs#L154-L235)).

### Hibiki Rust

The Rust demo app appends 12,000 zeros to the decoded file, then resamples if needed ([input preparation](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L51-L64)). It later uses `in_pcm_len / 1920` and exact 1,920-sample slices ([framing loop](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L103-L136)).

For an input already at 24 kHz, the appended zeros ensure that all original source samples, including a partial last frame, occur before the final processed full-frame boundary. This is an inference from the constants and loop, not a dedicated partial-frame or flush API. A trailing remainder after `floor((N + 12000) / 1920)` is still ignored.

There is also a source-rate caveat: the 12,000 zeros are appended **before** resampling. They represent 0.5 seconds only when the decoded input is already 24 kHz.

### Moshi Swift

The CLI loops in strides of 1,920, uses `min(start + 1920, pcm.count)`, and passes the possibly short final slice unchanged to `mimi.encodeStep` ([CLI loop](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiCLI/RunMoshi.swift#L111-L127)). The streaming convolution retains input that cannot form another stride and returns no output, but has no right-pad or flush path ([streaming convolution](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/Conv.swift#L235-L268)).

Consequently, the explicit behavior is “pass the short slice,” not “pad it.” The derived end result is that any codec state which has not emitted a frame is abandoned when the one-shot loop ends.

## 2. Silence-tail amount

The only Hibiki-specific public reference here that deliberately adds a tail is the Rust demo. Its literal policy is “append 12,000 zero samples” ([source](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L51-L64)), not “append six frames.”

At 24 kHz, 12,000 samples are 0.5 seconds, or 6.25 frames at 1,920 samples per frame. For original length `N`, the demo runs:

```text
floor((N + 12000) / 1920)
```

Compared with `floor(N / 1920)`, that adds six frames when `N mod 1920 < 1440`, otherwise seven. If `N` is frame-aligned, the app processes six full silent frames and ignores 480 trailing zeros. If `N` is not aligned, the first extra processed frame mixes the remaining source samples with zeros, followed by five or six all-zero frames. This paragraph is arithmetic derived from the cited constants, not an explicit comment in the source.

The checkpoint-era Python runner adds no tail ([finite loop](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L105-L140)). The later Python runner has a generic STT-only branch that pads by `audio_delay_seconds + 1.0` seconds, but that branch is inactive for the pinned Hibiki config ([branch](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L102-L111), [Hibiki config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L1-L48)). The Swift file and microphone paths have no model silence-tail loop ([file loop termination](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiCLI/RunMoshi.swift#L111-L161), [microphone model loop](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L653-L691)).

Therefore, “pad a partial frame and then add six silent frames” is deterministic and close to the Rust demo's nominal 0.5-second intent, but it differs from the Rust frame count for some input alignments.

## 3. Repeated finalization

None of the inspected implementations has a session-level `finish()`, `finalize()`, or codec `flush()` operation.

- Python's generator only accepts another source-token step and raises after `max_steps` ([generator limit](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/models/generate.py#L56-L60)); the runner is a one-shot function.
- Hibiki Rust has a one-shot `run` that writes the accumulated result after its finite loop ([loop and write](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L127-L183)).
- Swift's model protocol exposes `reset()` and a per-buffer callback, but no finalization operation ([protocol](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L400-L405)).

Repeated-finalization semantics are therefore completely unspecified by the references. Making the second call a safe no-op with an `already_finished` reason would be a new public lifecycle guarantee.

## 4. Incomplete delayed audio positions

The target schedule has a maximum acoustic delay of two frames ([pinned config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L33-L33)). In Python, a generation step writes each sampled codebook token at `step_idx - delay`; `last_audio_tokens()` exposes only the row at `step_idx - 1 - max_delay` and rejects padding or ungenerated values ([scheduler writes](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L107-L116), [completed-row lookup](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py#L138-L148)). The runner calls this once per source frame and then simply exits ([runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L141-L166)).

The official Moshi 0.5.2 Rust scheduler has the same pattern: delayed tokens are written into earlier rows, while `last_audio_tokens()` returns only `step_idx - acoustic_delay - 1` and returns `None` for an incomplete row ([writes](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/moshi-core/src/lm_generate_multistream.rs#L246-L275), [completed-row lookup](https://github.com/kyutai-labs/moshi/blob/0146d47f29726b134730acfd6f56f3575c4b236f/rust/moshi-core/src/lm_generate_multistream.rs#L319-L331)). The Hibiki app decodes only values returned by that method and does no later drain ([app loop](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L134-L168)).

Swift likewise writes sampled tokens at `stepIdx - delay` and exposes only `stepIdx - 1 - maxDelay`, returning no row containing padding and failing on an ungenerated value ([scheduler](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/LM.swift#L385-L451)). Its callers stop without a drain loop ([CLI](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiCLI/RunMoshi.swift#L115-L161), [app](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L653-L691)).

Derived consequence: after `N` LM steps with delay two, only `N - 2` complete target-audio rows can be exposed. The newest two schedule rows need future generation steps to fill their delayed codebooks. The runners neither decode these partial rows nor explicitly label, erase, or count them; their generator state simply goes out of scope or is reset. A package that discards and reports them would preserve the references' observable audio while making the loss explicit.

## EOS behavior

The versioned Hibiki paper describes a different intended protocol: training inserts a special input EOS on every source-audio token at the first frame after source speech and uses another text-stream EOS for the end of model speech ([training protocol](https://arxiv.org/html/2502.03382v2#S4.SS2.p4)); at inference, it says to send input EOS and keep sampling until the model produces its own EOS ([inference protocol](https://arxiv.org/html/2502.03382v2#S4.SS5.p1)).

The inspected public runners do not demonstrate that protocol end to end:

- Python runs a fixed number of PCM-derived steps, hides text ids 0 and 3 from printed output, and never breaks on them or injects an input EOS ([runner](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py#L128-L166)). Its generic config class calls audio id `audio_vocab_size - 2` the EOS token ([property](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/lm.py#L140-L146)), but the Hibiki runner does not use that property for finalization.
- Hibiki Rust labels text id 0 as EOP, but only suppresses ids 0 and 3 from output; it assigns either one as the next text input and continues its finite source loop ([configuration and loop](https://github.com/kyutai-labs/hibiki/blob/f1cf9293e35c1dceffbe60dd325bdd702bc8305e/hibiki-rs/src/gen.rs#L103-L168)).
- Swift defines `audioEOSToken()` as `audioVocabSize - 2` ([property](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/LM.swift#L70-L85)), but its Hibiki-labelled path merely hides text ids 0 and 3 and continues processing PCM ([app loop](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/Moshi/ContentView.swift#L653-L691)). Its generator stops only at `maxSteps` ([limit](https://github.com/kyutai-labs/moshi-swift/blob/df64ffdbe224e1ecb1ade1d254f347d379ed7f7a/MoshiLib/LM.swift#L385-L388)).

The paper establishes that learned EOS was intended, but these pinned runnable examples do not establish checkpoint-specific input injection, output stopping, or a reliable final drain contract. A silence-tail policy must therefore be named as fallback finalization, not described as paper-equivalent EOS completion.

## Implication for Q28–Q31

- **Q28:** Keep zero-padding the partial frame if the package promise is “do not silently lose source samples,” but record that this is a chosen package rule. It agrees most closely with the Rust demo's effective 24 kHz behavior, not with Python or Swift.
- **Q29:** Do not say that six frames reproduces the reference. The literal Rust-compatible rule is “append 12,000 zeros, process all new complete 1,920-sample frames, ignore a final all-zero remainder.” Combined with Q28's separate partial-frame padding, a fixed six-frame tail runs one more frame than Rust when the original nonempty remainder is below 1,440 samples; otherwise its frame count matches.
- **Q30:** Safe repeated `finish()` is sensible, but wholly new. Returning no duplicate output plus `already_finished` is compatible with the references because they define no competing behavior.
- **Q31:** Never expose a row until every delayed codebook is present. Explicitly discarding and counting the final incomplete rows is stricter and more observable than the references, whose runners leave those rows unreturned without a report.
