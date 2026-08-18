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

import mlx.core as mx
import sentencepiece
from mlx.utils import tree_map

from hibiki_mlx.artifacts.quantization import QuantizationSpec, quantize_linear_layers
from hibiki_mlx.models.lm import Lm, LmConfig
from hibiki_mlx.sampling import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "hibiki_mlx_swift/HibikiCore/Tests/HibikiCoreTests/Fixtures"

# A fixed, arbitrary frame: one text token and the sixteen audio streams the
# model consumes (source + target), plus the released "very_good" condition.
TEXT_TOKEN = 3
AUDIO_TOKENS = [(7 * i + 1) % 2048 for i in range(16)]
CONDITION = "very_good"

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
    # The tokenizer is shared across bundles; take it from whichever is present.
    for bundle_dir in bundles.values():
        if (bundle_dir / "config.json").exists():
            tokenizer_fixture(bundle_dir, FIXTURES / "tokenizer_decode.json")
            break


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    main()
