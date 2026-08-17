# Quantizing the Hibiki MLX artifact to Q8 and Q4

## Answer and scope

This note defines a compatible **weight-only LM** conversion for the pinned
`kyutai/hibiki-1b-mlx-bf16` artifact used by this repository.  It does not
change the Mimi codec.  It is based on the installed MLX 0.26.5 API and the
historical Kyutai MLX runner's Q8/Q4 loader convention.

| Variant | MLX bits | Group size | Use |
| --- | ---: | ---: | --- |
| Q8 | 8 | 64 | First quality/performance baseline |
| Q4 | 4 | 32 | Smaller artifact; requires separate quality evidence |

Those bit/group pairs are the ones Kyutai's MLX runner selects for
`.q8.safetensors` and `.q4.safetensors`, respectively ([reference loader](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L80-L86)).

The conversion target is the local `Lm`, not the whole `LoadedModel` and not
Mimi. MLX's `nn.quantize()` replaces eligible leaf modules **in place** with
quantized modules ([implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/quantized.py#L11-L57)).  A converted
`QuantizedLinear` stores packed `weight`, plus `scales` and `biases`, and runs
with `mx.quantized_matmul` ([implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/quantized.py#L145-L230)). It is therefore a different parameter schema from the BF16 `Linear`.

## The selection required by this artifact

Do **not** call `nn.quantize(lm)` with its default predicate. By default MLX
attempts every leaf with `to_quantized`, including `Linear` and `Embedding`
([implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/quantized.py#L19-L56), [linear](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/linear.py#L73-L75), [embedding](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/embedding.py#L42-L44)). MLX 0.26 only accepts a two-dimensional quantized matrix whose dimensions are multiples of 32, and whose column count is divisible by the group size ([`mx.quantize` documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)).

The pinned config has 48,000 text tokens, 2,048 audio tokens, a 2,048-wide
Temporal Transformer, and a 1,024-wide Depth Transformer
([config](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json#L1-L48)). The local model adds one padding entry to the text/audio input vocabularies. Consequently, these leaf matrices are not legal MLX 0.26 quantization inputs:

- `ScaledEmbedding` instances: `text_emb` and the first depth embedding are
  48,001 rows; audio and remaining depth embeddings are 2,049 rows. They also
  preserve the `-1`-to-zero behavior supplied by `ScaledEmbedding`, which a
  plain `QuantizedEmbedding` would not retain.
- The description conditioner embedding is 32 × 16 and its output projection
  is 2,048 × 16. Their 16-wide columns violate both Q8's group of 64 and Q4's
  group of 32.
- Mimi uses repository-local convolution modules and raw codebook arrays,
  rather than MLX `Linear`/`Embedding` leaves. It remains in its released
  float32 format. ([Mimi architecture](../../hibiki_mlx/hibiki_mlx/models/mimi.py), [local convolution](../../hibiki_mlx/hibiki_mlx/modules/conv.py))

For the currently pinned artifact, select only compatible `nn.Linear` leaves:

```python
import mlx.nn as nn


def quantizable_linear(_path: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    weight = module.weight
    return (
        weight.ndim == 2
        and all(dimension % 32 == 0 for dimension in weight.shape)
        and weight.shape[-1] % GROUP_SIZE == 0
    )
```

This predicate selects the Temporal/Depth Transformer projections, the
per-depth-slice projections, and `text_linear`; it excludes embeddings and
the small conditioner projection. Direct inspection of the pinned safetensors
finds 273 such matrices: 2.925 GiB of its 3.353 GiB payload (87.3%). The
remaining 0.427 GiB stays BF16. These are artifact-header measurements, not a
prediction of allocator memory.

## One-time conversion procedure

Perform conversion on a Mac that can run the `hibiki` Conda environment and
has enough free unified memory and disk for both the 3.4-GB source LM and the
new artifact. Keep the pinned BF16 artifact immutable.

1. Read `config.json`, build `Lm(LmConfig.from_config_dict(config))`, call
   `lm.set_dtype(mx.bfloat16)`, and strictly load the original
   `hibiki-mlx-dc2cf5a5@80.safetensors`. This matches the normal repository
   loading order ([current loader](../../hibiki_mlx/hibiki_mlx/inference.py)).
2. Set `BITS, GROUP_SIZE = (8, 64)` for Q8, or `(4, 32)` for Q4. Run the
   predicate above through `nn.quantize` and force the new tensors to finish
   before saving:

   ```python
   nn.quantize(
       lm,
       bits=BITS,
       group_size=GROUP_SIZE,
       class_predicate=quantizable_linear,
   )
   mx.eval(lm.parameters())
   lm.save_weights(output_path)  # output_path ends in .safetensors
   ```

   `Module.save_weights()` dispatches a `.safetensors` destination to
   `mx.save_safetensors` ([MLX implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/base.py#L209-L224)).
3. Build a separate artifact bundle for each variant. It needs the same
   `config.json`, Mimi safetensors, and SentencePiece model as the source
   bundle, plus the new LM file (for example,
   `hibiki-mlx-dc2cf5a5@80.q4.safetensors`). The config's `moshi_name` must
   name that new file; `mimi_name` and `tokenizer_name` remain unchanged. The
   current bundle resolver reads exactly those fields
   ([implementation](../../hibiki_mlx/hibiki_mlx/inference.py)). Symlinking
   the unchanged Mimi/tokenizer into the variant directory is sufficient for
   local use; distributing a standalone bundle requires including them.
4. Write a small manifest alongside each new bundle that records: source
   repository/revision, source and output SHA-256, MLX version `0.26.5`, bits,
   group size, and the predicate version (for example,
   `linear-shape-v1`). Neither safetensors nor `save_weights()` preserves the
   `QuantizedLinear.bits`/`group_size` Python attributes as model metadata.

The conversion script must load BF16 first, then replace modules, then save.
Quantizing an empty model before loading the BF16 source does not work because
the quantized module expects packed weights plus `scales`/`biases`, while the
source has a single normal `weight`. Conversely, a quantized file must not be
loaded into the unmodified BF16 module tree. MLX strict loading checks both
parameter names and shapes ([MLX implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/base.py#L123-L207)).

## Reload contract

The current `hibiki_mlx.load_model()` only constructs the BF16 structure, so
it cannot load either output yet. Add an explicit quantization selection (or
read the variant manifest) and use this order for a Q4/Q8 bundle:

```python
lm = Lm(lm_config)
lm.set_dtype(mx.bfloat16)
nn.quantize(
    lm,
    bits=BITS,
    group_size=GROUP_SIZE,
    class_predicate=quantizable_linear,
)
lm.load_weights(str(quantized_lm_path), strict=True)
```

The values and the predicate must be exactly the same as during conversion.
Filename suffix alone is not a safe contract for this repository: although
the historical runner uses it, this narrower Hibiki selection intentionally
differs from the runner's default quantization.

## Validation and limitations

- Q8 and Q4 are approximate weight-only models. Quantized projections alter
  logits, so sampled text tokens and generated PCM are not expected to be
  bit-exact with BF16. Treat the BF16 bundle as the compatibility baseline.
- Validate Q8 first, then Q4, using deterministic seeds and the existing
  short-form French fixture plus longer and more varied speech. Compare text,
  intelligibility, voice quality, real-time step rate, peak MLX active/cache
  memory, and process RSS. A successful strict load alone does not establish
  translation quality.
- The on-disk reduction applies principally to the 87.3% of LM payload listed
  above. Mimi's roughly 367-MB released float32 file, embeddings, norms,
  caches, activations, decoded-audio queues, and allocator cache remain. Do
  not derive an iPhone memory budget from file size alone.
- Creating the quantized weights may temporarily need more memory than either
  finished model because the BF16 source and quantized arrays coexist until
  evaluation/save completes. Never overwrite the source safetensors in place.
- This is the MLX Python packed-weight layout. A native MLX Swift iPhone
  runner must explicitly construct compatible quantized linear layers and
  consume the same packed weights/scales/biases, or perform its own conversion;
  that compatibility is not established by this Python procedure.

## Sources

- Apple MLX 0.26.5: [`nn.quantize` and quantized layer implementation](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/quantized.py), [`Linear`](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/linear.py), [`Embedding`](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/embedding.py), and [`Module.load_weights` / `save_weights`](https://github.com/ml-explore/mlx/blob/v0.26.5/python/mlx/nn/layers/base.py).
- Kyutai Moshi's historical [MLX Q4/Q8 load convention](https://github.com/kyutai-labs/moshi/blob/dd6b9fffd613e5a2c64166a7ec09b121be09877b/moshi_mlx/moshi_mlx/run_inference.py#L73-L88).
- Kyutai's pinned [Hibiki BF16 configuration](https://huggingface.co/kyutai/hibiki-1b-mlx-bf16/blob/b3d6291f3dcf7954e1a502e4d66f32e3556f17ae/config.json).
