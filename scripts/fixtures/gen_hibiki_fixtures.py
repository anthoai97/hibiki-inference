"""Generate MLX Swift parity fixtures from the Python Hibiki reference.

The native Swift port is verified on the macOS CPU backend in float32, so the
fixtures are produced the same way: build the model, quantize the LM linears
when the bundle declares it, load the released weights, then widen every
non-packed parameter to float32 (the packed ``uint32`` quantized weights stay
as they are). Sampling is greedy so the fixtures are deterministic and free of
any Swift/Python RNG mismatch.

Run (from the repo root)::

    PYTHONPATH=hibiki_mlx conda run -n hibiki \
        python scripts/fixtures/gen_hibiki_fixtures.py

It writes into ``hibiki_mlx_swift/HibikiCore/Tests/HibikiCoreTests/Fixtures``:

* ``sample_step_bf16.safetensors`` / ``sample_step_q8.safetensors`` -- one warm
  generation step (temporal state, sampled text token, sampled audio tokens).
* ``tokenizer_decode.json`` -- id sequences and their ``sp.decode`` output.
"""

from __future__ import annotations

import json
from pathlib import Path

import math

import mlx.core as mx
import sentencepiece
from mlx.utils import tree_map

from hibiki_mlx.artifacts.quantization import QuantizationSpec, quantize_linear_layers
from hibiki_mlx.generate import LmGen
from hibiki_mlx.models.lm import Lm, LmConfig
from hibiki_mlx.models.mimi import Mimi, mimi_202407, remap_released_weights
from hibiki_mlx.sampling import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "hibiki_mlx_swift/HibikiCore/Tests/HibikiCoreTests/Fixtures"

# A fixed, arbitrary frame: one text token and the sixteen audio streams the
# model consumes (source + target), plus the released "very_good" condition.
TEXT_TOKEN = 3
AUDIO_TOKENS = [(7 * i + 1) % 2048 for i in range(16)]
CONDITION = "very_good"

# Number of 80 ms source frames the Mimi round-trip fixture streams.
MIMI_FRAMES = 8

# Number of frames the delayed-scheduler fixture drives.
LMGEN_FRAMES = 12

# Decode cases exercised by the Swift tokenizer parity test.
DECODE_CASES = [
    [70, 100, 200, 300, 400],
    [1000, 2000, 3000, 42, 5],
    [10, 10, 10, 500, 12345],
]


def load_lm_cpu_f32(bundle_dir: Path) -> tuple[Lm, LmConfig, dict]:
    """Load one bundle's LM exactly as the Swift CPU path does."""
    config = json.loads((bundle_dir / "config.json").read_text())
    lm_config = LmConfig.from_config_dict(config)
    lm = Lm(lm_config)
    spec = QuantizationSpec.from_config(config)
    if spec is not None:
        quantize_linear_layers(lm, spec)
    lm.load_weights(str(bundle_dir / config["moshi_name"]), strict=True)
    lm.update(tree_map(lambda x: x if x.dtype == mx.uint32 else x.astype(mx.float32), lm.parameters()))
    mx.eval(lm.parameters())
    return lm, lm_config, config


def sample_step_fixture(bundle_dir: Path, out_path: Path) -> None:
    lm, lm_config, _ = load_lm_cpu_f32(bundle_dir)
    greedy = Sampler(temp=0)

    text_ids = mx.array([[TEXT_TOKEN]], dtype=mx.int32)
    audio_ids = [mx.array([[t]], dtype=mx.int32) for t in AUDIO_TOKENS]
    condition = lm.condition_provider.condition_tensor("description", CONDITION)

    # The temporal state, captured the way sample_step computes it.
    xs = lm.text_emb(text_ids)
    for token_ids, embedding in zip(audio_ids, lm.audio_embs):
        xs = xs + embedding(token_ids)
    xs = xs + mx.expand_dims(condition.tensor, axis=1)
    transformer_out = lm.out_norm(lm.transformer(xs, cache=lm.make_transformer_cache()))

    text_token = greedy(lm.text_linear(transformer_out))
    audio_tokens = lm.depformer.sample(
        transformer_out, text_token, lm.make_depformer_cache(), greedy
    )
    mx.eval(transformer_out, text_token, audio_tokens)

    mx.save_safetensors(
        str(out_path),
        {
            "text_token_ids": text_ids,
            "audio_token_ids": mx.concatenate(audio_ids, axis=0).astype(mx.int32),  # [16, 1]
            "transformer_out": transformer_out.astype(mx.float32),
            "text_token": text_token.astype(mx.int32),
            "audio_tokens": audio_tokens.astype(mx.int32),
        },
    )
    print(f"wrote {out_path.name}: text_token={text_token.tolist()} audio={audio_tokens.squeeze(-1).tolist()}")


def lmgen_fixture(bundle_dir: Path, out_path: Path) -> None:
    lm, lm_config, _ = load_lm_cpu_f32(bundle_dir)
    greedy = Sampler(temp=0)
    gen = LmGen(lm, greedy, greedy)
    provider = lm.condition_provider
    condition = provider.condition_tensor("description", CONDITION) if provider is not None else None

    source_cb = lm_config.source_codebooks
    source = mx.array(
        [[(3 * i + 5 * c + 1) % 2048 for c in range(source_cb)] for i in range(LMGEN_FRAMES)],
        dtype=mx.int32,
    )  # [frames, source_codebooks]

    texts: list[int] = []
    audio_frames = []
    for i in range(LMGEN_FRAMES):
        token = gen.step(source[i : i + 1], condition)
        texts.append(int(token.squeeze().item()))
        frame = gen.last_audio_tokens()
        if frame is not None:
            audio_frames.append(frame)  # [1, target_codebooks]

    text = mx.array(texts, dtype=mx.int32)  # [frames]
    audio = (
        mx.concatenate(audio_frames, axis=0)
        if audio_frames
        else mx.zeros((0, lm_config.target_codebooks), dtype=mx.int32)
    )  # [ready_frames, target_codebooks]
    mx.eval(text, audio)

    mx.save_safetensors(
        str(out_path),
        {"source_tokens": source, "text_tokens": text, "audio_frames": audio},
    )
    print(f"wrote {out_path.name}: text {text.shape}, audio_frames {audio.shape}")


def load_mimi_cpu_f32(bundle_dir: Path) -> tuple[Mimi, int]:
    """Load one bundle's Mimi codec exactly as the Swift CPU path does."""
    config = json.loads((bundle_dir / "config.json").read_text())
    lm_config = LmConfig.from_config_dict(config)
    codebooks = lm_config.target_codebooks
    mimi = Mimi(mimi_202407(codebooks))
    codec_weights, _ = remap_released_weights(mx.load(str(bundle_dir / config["mimi_name"])), codebooks=codebooks)
    mimi.load_weights(list(codec_weights.items()), strict=True)
    mimi.refresh_derived_state()
    mx.eval(mimi.parameters())
    return mimi, codebooks


def mimi_fixture(bundle_dir: Path, out_path: Path) -> None:
    mimi, _ = load_mimi_cpu_f32(bundle_dir)
    frame_size = mimi.cfg.frame_size

    # A fixed, deterministic 220 Hz tone, saved into the fixture so the Swift
    # test streams identical samples.
    total = MIMI_FRAMES * frame_size
    samples = [0.1 * math.sin(2.0 * math.pi * 220.0 * i / mimi.cfg.sample_rate) for i in range(total)]
    input_pcm = mx.array(samples, dtype=mx.float32).reshape(1, 1, total)

    mimi.reset_state()
    encoder_cache = mimi.make_encoder_cache()
    codes = []
    for index in range(MIMI_FRAMES):
        frame = input_pcm[:, :, index * frame_size : (index + 1) * frame_size]
        step = mimi.encode_step(frame, encoder_cache)
        if step.shape[-1] > 0:
            codes.append(step)
    all_codes = mx.concatenate(codes, axis=-1)  # [1, nq, T]

    decoder_cache = mimi.make_decoder_cache()
    pcm = []
    for index in range(all_codes.shape[-1]):
        pcm.append(mimi.decode_step(all_codes[:, :, index : index + 1], decoder_cache))
    output_pcm = mx.concatenate(pcm, axis=-1)  # [1, 1, T']
    mx.eval(all_codes, output_pcm)

    mx.save_safetensors(
        str(out_path),
        {
            "input_pcm": input_pcm,
            "codes": all_codes.astype(mx.int32),
            "output_pcm": output_pcm.astype(mx.float32),
        },
    )
    print(f"wrote {out_path.name}: codes {all_codes.shape}, output_pcm {output_pcm.shape}")


def tokenizer_fixture(bundle_dir: Path, out_path: Path) -> None:
    config = json.loads((bundle_dir / "config.json").read_text())
    sp = sentencepiece.SentencePieceProcessor(str(bundle_dir / config["tokenizer_name"]))
    cases = [{"ids": ids, "text": sp.decode(ids)} for ids in DECODE_CASES]
    out_path.write_text(json.dumps({"vocab_size": sp.vocab_size(), "cases": cases}, ensure_ascii=False, indent=2))
    print(f"wrote {out_path.name}: {len(cases)} cases, vocab {sp.vocab_size()}")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    bundles = {
        "bf16": REPO_ROOT / "artifacts/hibiki-1b-mlx-bf16",
        "q8": REPO_ROOT / "artifacts/hibiki-1b-mlx-q8",
    }
    for tag, bundle_dir in bundles.items():
        if not (bundle_dir / "config.json").exists():
            print(f"skipping {tag}: {bundle_dir} not present")
            continue
        sample_step_fixture(bundle_dir, FIXTURES / f"sample_step_{tag}.safetensors")
        lmgen_fixture(bundle_dir, FIXTURES / f"lmgen_{tag}.safetensors")
    # The Mimi codec and tokenizer are shared across bundles; take them from
    # whichever bundle is present.
    for bundle_dir in bundles.values():
        if (bundle_dir / "config.json").exists():
            mimi_fixture(bundle_dir, FIXTURES / "mimi_roundtrip.safetensors")
            tokenizer_fixture(bundle_dir, FIXTURES / "tokenizer_decode.json")
            break


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    main()
