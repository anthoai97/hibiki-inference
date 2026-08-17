"""The parameters each set of released weights must contain.

The expected parameter set is *derived* -- the language model's from the bundle's
own `config.json`, the codec's from the built-in `MimiSpec` contract below. It is
not a recorded dump of one release, so an artifact whose weights disagree with
its configuration is rejected even when the revision is unknown and hash trust
has been waived.

`tests/data/weight_manifest_*.json`, captured off the released artifacts,
is the independent oracle that keeps these derivations honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._safetensors import TensorManifest, read_tensor_manifest
from .errors import IncompatibleArtifactError
from .spec import SOURCE_FRAME_SAMPLES, ModelSpec

# The released weights are BF16 for the language model and float32 for the codec.
LM_DTYPE = "BF16"
MIMI_DTYPE = "F32"

# How many mismatches to name before truncating an error message.
_MAX_REPORTED = 5


@dataclass(frozen=True, slots=True)
class MimiSpec:
    """The Mimi codec architecture this implementation targets.

    Unlike the language model, the codec's shape is not described by the
    bundle's `config.json`; it is fixed by the released Mimi checkpoint. Holding
    it as an explicit contract is what lets a repackaged or differently
    configured codec be rejected rather than half-loaded.
    """

    sample_rate: int = 24000
    channels: int = 1
    dimension: int = 512
    n_filters: int = 64
    # Decoder order. The encoder applies these in reverse.
    ratios: tuple[int, ...] = (8, 6, 5, 4)
    kernel_size: int = 7
    residual_kernel_size: int = 3
    last_kernel_size: int = 3
    compress: int = 2
    transformer_dim: int = 512
    transformer_layers: int = 8
    transformer_heads: int = 8
    transformer_feedforward: int = 2048
    # The codec resamples by this stride either side of its bottleneck.
    resample_stride: int = 2
    quantizer_dim: int = 256
    semantic_codebooks: int = 1
    acoustic_codebooks: int = 31

    @property
    def samples_per_source_frame(self) -> int:
        """PCM samples one pass of the encoder stages consumes."""
        stride = self.resample_stride
        for ratio in self.ratios:
            stride *= ratio
        return stride

    @property
    def total_codebooks(self) -> int:
        return self.semantic_codebooks + self.acoustic_codebooks


DEFAULT_MIMI_SPEC = MimiSpec()


@dataclass(slots=True)
class _ManifestBuilder:
    """Accumulates expected parameters, all sharing one dtype."""

    dtype: str
    parameters: TensorManifest = field(default_factory=dict)

    def add(self, name: str, *shape: int) -> None:
        self.parameters[name] = (self.dtype, shape)


def expected_lm_parameters(spec: ModelSpec) -> TensorManifest:
    """Derive the Hibiki language model's parameters from a validated config."""
    out = _ManifestBuilder(LM_DTYPE)

    out.add("text_emb.weight", spec.text_input_card, spec.dim)
    out.add("text_linear.weight", spec.text_card, spec.dim)
    out.add("out_norm.weight", spec.dim)

    # One embedding table per temporal audio stream: `dep_q` target streams
    # followed by `dep_q` teacher-forced source streams.
    for stream in range(spec.n_q):
        out.add(f"audio_embs.{stream}.weight", spec.audio_input_card, spec.dim)

    conditioner = "condition_provider.conditioners.description"
    out.add(f"{conditioner}.embed.weight", spec.condition_bins + 1, spec.condition_dim)
    out.add(f"{conditioner}.learnt_padding", 1, 1, spec.dim)
    out.add(f"{conditioner}.output_proj.weight", spec.dim, spec.condition_dim)

    for layer in range(spec.num_layers):
        _add_gated_attention_layer(
            out,
            prefix=f"transformer.layers.{layer}",
            dim=spec.dim,
            gated_width=spec.gated_width,
        )

    depth_dim = spec.depformer_dim
    for step in range(spec.dep_q):
        prefix = f"depformer.slices.{step}"
        # Depth step 0 is conditioned on the text token sampled this frame; every
        # later step on the audio code the previous step produced.
        rows = spec.text_input_card if step == 0 else spec.audio_input_card
        out.add(f"{prefix}.emb.weight", rows, depth_dim)
        out.add(f"{prefix}.linear_in.weight", depth_dim, spec.dim)
        out.add(f"{prefix}.linear_out.weight", spec.card, depth_dim)
        for layer in range(spec.depformer_num_layers):
            _add_gated_attention_layer(
                out,
                prefix=f"{prefix}.transformer.layers.{layer}",
                dim=depth_dim,
                gated_width=spec.depformer_gated_width,
            )

    return out.parameters


def _add_gated_attention_layer(
    out: _ManifestBuilder, *, prefix: str, dim: int, gated_width: int
) -> None:
    """A pre-norm attention layer with a fused QKV projection and gated feed-forward."""
    out.add(f"{prefix}.self_attn.in_proj.weight", 3 * dim, dim)
    out.add(f"{prefix}.self_attn.out_proj.weight", dim, dim)
    out.add(f"{prefix}.gating.linear_in.weight", 2 * gated_width, dim)
    out.add(f"{prefix}.gating.linear_out.weight", dim, gated_width)
    out.add(f"{prefix}.norm1.weight", dim)
    out.add(f"{prefix}.norm2.weight", dim)


def expected_mimi_parameters(spec: ModelSpec, mimi: MimiSpec = DEFAULT_MIMI_SPEC) -> TensorManifest:
    """Derive the Mimi codec's parameters, cross-checked against the config."""
    _check_codec_contract_is_self_consistent(spec, mimi)
    out = _ManifestBuilder(MIMI_DTYPE)

    _add_encoder(out, mimi)
    _add_decoder(out, mimi)
    for prefix in ("encoder_transformer", "decoder_transformer"):
        _add_codec_transformer(out, mimi, prefix)

    # Bottleneck resampling: a strided convolution down, a depthwise transposed
    # convolution back up.
    kernel = mimi.resample_stride * 2
    out.add("downsample.conv.conv.conv.weight", mimi.dimension, mimi.dimension, kernel)
    out.add("upsample.convtr.convtr.convtr.weight", mimi.dimension, 1, kernel)

    _add_quantizer(out, spec, mimi)
    return out.parameters


def _check_codec_contract_is_self_consistent(spec: ModelSpec, mimi: MimiSpec) -> None:
    """Guard the built-in codec contract, not the artifacts.

    Both values below are constants of this implementation rather than anything
    read from a bundle, so this cannot fail on artifact drift. It exists because
    `MimiSpec` is what later codec work will edit: changing a ratio or the
    resample stride would silently move the frame off the 12.5 Hz model clock,
    and asking for more codebooks than the codec carries would silently truncate.
    """
    if mimi.samples_per_source_frame != SOURCE_FRAME_SAMPLES:
        raise IncompatibleArtifactError(
            f"The codec contract compresses {mimi.samples_per_source_frame} samples "
            f"per frame, but a source frame is {SOURCE_FRAME_SAMPLES} samples."
        )
    if spec.dep_q > mimi.total_codebooks:
        raise IncompatibleArtifactError(
            f"The configuration generates {spec.dep_q} codebooks, but the codec "
            f"contract provides only {mimi.total_codebooks}."
        )


def _add_encoder(out: _ManifestBuilder, mimi: MimiSpec) -> None:
    """SEANet encoder: an input convolution, then one stage per ratio."""
    out.add("encoder.model.0.conv.conv.weight", mimi.n_filters, mimi.channels, mimi.kernel_size)
    out.add("encoder.model.0.conv.conv.bias", mimi.n_filters)

    index, width = 1, mimi.n_filters
    for ratio in reversed(mimi.ratios):
        _add_residual_block(out, f"encoder.model.{index}", mimi, width)
        # Downsampling convolution: stride `ratio`, kernel twice the stride.
        out.add(f"encoder.model.{index + 2}.conv.conv.weight", width * 2, width, ratio * 2)
        out.add(f"encoder.model.{index + 2}.conv.conv.bias", width * 2)
        index, width = index + 3, width * 2

    final = f"encoder.model.{index + 1}.conv.conv"
    out.add(f"{final}.weight", mimi.dimension, width, mimi.last_kernel_size)
    out.add(f"{final}.bias", mimi.dimension)


def _add_decoder(out: _ManifestBuilder, mimi: MimiSpec) -> None:
    """SEANet decoder: the encoder mirrored, with transposed convolutions."""
    width = mimi.n_filters * 2 ** len(mimi.ratios)
    out.add("decoder.model.0.conv.conv.weight", width, mimi.dimension, mimi.kernel_size)
    out.add("decoder.model.0.conv.conv.bias", width)

    index = 2
    for ratio in mimi.ratios:
        upsampled = width // 2
        out.add(f"decoder.model.{index}.convtr.convtr.weight", width, upsampled, ratio * 2)
        out.add(f"decoder.model.{index}.convtr.convtr.bias", upsampled)
        _add_residual_block(out, f"decoder.model.{index + 1}", mimi, upsampled)
        index, width = index + 3, upsampled

    out.add(f"decoder.model.{index}.conv.conv.weight", mimi.channels, width, mimi.last_kernel_size)
    out.add(f"decoder.model.{index}.conv.conv.bias", mimi.channels)


def _add_residual_block(out: _ManifestBuilder, prefix: str, mimi: MimiSpec, width: int) -> None:
    """A residual block that compresses the channel count and restores it."""
    squeezed = width // mimi.compress
    out.add(f"{prefix}.block.1.conv.conv.weight", squeezed, width, mimi.residual_kernel_size)
    out.add(f"{prefix}.block.1.conv.conv.bias", squeezed)
    out.add(f"{prefix}.block.3.conv.conv.weight", width, squeezed, 1)
    out.add(f"{prefix}.block.3.conv.conv.bias", width)


def _add_codec_transformer(out: _ManifestBuilder, mimi: MimiSpec, prefix: str) -> None:
    """The codec's projected Transformer: layer-normed, layer-scaled, ungated."""
    dim, feedforward = mimi.transformer_dim, mimi.transformer_feedforward
    for layer in range(mimi.transformer_layers):
        layer_prefix = f"{prefix}.transformer.layers.{layer}"
        out.add(f"{layer_prefix}.self_attn.in_proj_weight", 3 * dim, dim)
        out.add(f"{layer_prefix}.self_attn.out_proj.weight", dim, dim)
        out.add(f"{layer_prefix}.linear1.weight", feedforward, dim)
        out.add(f"{layer_prefix}.linear2.weight", dim, feedforward)
        for norm in ("norm1", "norm2"):
            out.add(f"{layer_prefix}.{norm}.weight", dim)
            out.add(f"{layer_prefix}.{norm}.bias", dim)
        for scale in ("layer_scale_1", "layer_scale_2"):
            out.add(f"{layer_prefix}.{scale}.scale", dim)


def _add_quantizer(out: _ManifestBuilder, spec: ModelSpec, mimi: MimiSpec) -> None:
    """Split residual vector quantizer: one semantic codebook, then the acoustic rest.

    The release carries more codebooks than Hibiki reads; the surplus is part of
    the artifact, so it is required rather than treated as unexpected.
    """
    groups = (
        ("rvq_first", mimi.semantic_codebooks),
        ("rvq_rest", mimi.acoustic_codebooks),
    )
    for group, codebooks in groups:
        out.add(f"quantizer.{group}.input_proj.weight", mimi.quantizer_dim, mimi.dimension, 1)
        out.add(f"quantizer.{group}.output_proj.weight", mimi.dimension, mimi.quantizer_dim, 1)
        for codebook in range(codebooks):
            prefix = f"quantizer.{group}.vq.layers.{codebook}._codebook"
            out.add(f"{prefix}._initialized", 1)
            out.add(f"{prefix}.cluster_usage", spec.card)
            out.add(f"{prefix}.embedding_sum", spec.card, mimi.quantizer_dim)


def verify_weights(path: Path, expected: TensorManifest, *, role: str, described_by: str) -> None:
    """Reject weights whose parameter names, dtypes, or shapes are not `expected`.

    `role` names the weights in the error; `described_by` names whatever defined
    the expectation, so the caller is told which authority to go and look at.
    """
    actual = read_tensor_manifest(path)

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = [
        (name, expected[name], actual[name])
        for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    ]
    if not (missing or unexpected or mismatched):
        return

    problems: list[str] = []
    if missing:
        problems.append(f"missing {len(missing)}: {_summarize(missing)}")
    if unexpected:
        problems.append(f"unexpected {len(unexpected)}: {_summarize(unexpected)}")
    if mismatched:
        described = [
            f"{name} is {actual_entry[0]}{list(actual_entry[1])} "
            f"but should be {expected_entry[0]}{list(expected_entry[1])}"
            for name, expected_entry, actual_entry in mismatched
        ]
        problems.append(f"wrong shape or dtype {len(mismatched)}: {_summarize(described)}")

    raise IncompatibleArtifactError(
        f"The {role} weights at {path} do not match the architecture "
        f"{described_by} describes -- {'; '.join(problems)}. The bundle cannot "
        f"drive this implementation."
    )


def _summarize(items: list[str]) -> str:
    shown = ", ".join(items[:_MAX_REPORTED])
    remaining = len(items) - _MAX_REPORTED
    return f"{shown}, and {remaining} more" if remaining > 0 else shown
