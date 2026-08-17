"""The architecture described by a bundle's `config.json`.

`ModelSpec` is the parsed, validated form of the released configuration. It is
the single place that decides what "compatible with this implementation" means,
so later stages can build modules from it without re-reading raw JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import IncompatibleArtifactError

CONFIG_FILENAME = "config.json"

# Conditioner labels the released `description` lookup table is trained on.
CONDITION_LABELS = ("very_bad", "bad", "neutral", "good", "very_good")

# A source frame is exactly 1,920 samples of 24 kHz mono PCM, or 80 ms on the
# model timeline. It comes from the 12.5 Hz model clock, not from `config.json`.
SOURCE_FRAME_SAMPLES = 1920


def _require(field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} has {field}={actual!r}, but this implementation "
            f"requires {field}={expected!r}. The bundle is not the supported "
            f"Hibiki 1B architecture."
        )


def _field(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise IncompatibleArtifactError(f"{CONFIG_FILENAME} is missing the required key {key!r}.")
    return raw[key]


def _mapping(value: object, described: str) -> dict[str, Any]:
    """Require a JSON object, so malformed input fails as a public error."""
    if not isinstance(value, dict):
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} has {described} of type {type(value).__name__}, "
            f"but an object is required."
        )
    return value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Validated architecture of one artifact bundle."""

    # Artifact names the configuration declares for its own siblings.
    lm_weights_name: str
    mimi_weights_name: str
    tokenizer_name: str

    # Temporal Transformer.
    dim: int
    num_layers: int
    num_heads: int
    context: int
    max_period: int
    hidden_scale: float

    # Streams and vocabularies.
    text_card: int
    text_padding_id: int
    n_q: int
    dep_q: int
    card: int
    delays: tuple[int, ...]

    # Depth Transformer.
    depformer_dim: int
    depformer_num_layers: int
    depformer_num_heads: int
    depformer_dim_feedforward: int
    depformer_context: int

    # Conditioning.
    condition_dim: int
    condition_bins: int

    @property
    def text_input_card(self) -> int:
        """Text embedding rows: every output token plus the input-only start id."""
        return self.text_card + 1

    @property
    def audio_input_card(self) -> int:
        """Audio embedding rows: every code plus the language-model padding id."""
        return self.card + 1

    @property
    def gated_width(self) -> int:
        """Width of one branch of the Temporal Transformer's gated feed-forward."""
        return _gated_width(round(self.hidden_scale * self.dim))

    @property
    def depformer_gated_width(self) -> int:
        """Width of one branch of the Depth Transformer's gated feed-forward."""
        return _gated_width(self.depformer_dim_feedforward)


def _gated_width(feedforward_dim: int) -> int:
    """Gated feed-forward branches are two thirds of the nominal width."""
    width, remainder = divmod(2 * feedforward_dim, 3)
    if remainder:
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} implies a gated feed-forward width of "
            f"2*{feedforward_dim}/3, which is not a whole number."
        )
    return width


def parse_model_spec(raw: object) -> ModelSpec:
    """Parse `config.json` and reject any drift from the supported architecture.

    Every check here guards something a later stage would otherwise discover as
    a shape mismatch, a silently wrong schedule, or wrong audio. Drift in
    cardinalities, the codebook split, the delay schedule, attention context,
    layer counts, or conditioner labels is rejected before anything is allocated.
    """
    raw = _mapping(raw, "a top-level value")

    # Stream layout: 16 temporal audio streams are 8 target plus 8 source, not
    # 16 generated codebooks, so `dep_q` is half of `n_q` by construction.
    n_q = _field(raw, "n_q")
    dep_q = _field(raw, "dep_q")
    _require("n_q", n_q, 16)
    _require("dep_q", dep_q, 8)

    # One text stream plus every audio stream. The first codebook of each audio
    # stream is semantic and undelayed; its seven acoustic codebooks lag by two
    # 80 ms frames.
    expected_delays = (0, *((0,) + (2,) * (dep_q - 1)) * 2)
    delays = tuple(_field(raw, "delays"))
    if delays != expected_delays:
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} has delays={list(delays)}, but this "
            f"implementation requires {list(expected_delays)}: one undelayed "
            f"text stream, then {dep_q} target and {dep_q} source streams whose "
            f"first codebook is undelayed and whose rest lag by two frames."
        )

    _require("card", _field(raw, "card"), 2048)
    _require("text_card", _field(raw, "text_card"), 48000)
    _require("existing_text_padding_id", _field(raw, "existing_text_padding_id"), 3)

    _require("dim", _field(raw, "dim"), 2048)
    _require("num_layers", _field(raw, "num_layers"), 16)
    _require("num_heads", _field(raw, "num_heads"), 16)
    _require("context", _field(raw, "context"), 500)
    _require("max_period", _field(raw, "max_period"), 100000)
    _require("hidden_scale", _field(raw, "hidden_scale"), 4.125)
    _require("causal", _field(raw, "causal"), True)
    _require("gating", _field(raw, "gating"), "silu")
    _require("norm", _field(raw, "norm"), "rms_norm_f32")
    _require("positional_embedding", _field(raw, "positional_embedding"), "rope")
    _require("layer_scale", raw.get("layer_scale"), None)

    _require("depformer_dim", _field(raw, "depformer_dim"), 1024)
    _require("depformer_num_layers", _field(raw, "depformer_num_layers"), 6)
    _require("depformer_num_heads", _field(raw, "depformer_num_heads"), 16)
    _require("depformer_dim_feedforward", _field(raw, "depformer_dim_feedforward"), 4224)
    _require("depformer_context", _field(raw, "depformer_context"), 16)
    _require("depformer_causal", _field(raw, "depformer_causal"), True)
    _require("depformer_gating", _field(raw, "depformer_gating"), "silu")
    _require("depformer_pos_emb", _field(raw, "depformer_pos_emb"), "none")
    _require("depformer_multi_linear", _field(raw, "depformer_multi_linear"), True)
    _require("depformer_weights_per_step", _field(raw, "depformer_weights_per_step"), True)
    _require("depformer_layer_scale", raw.get("depformer_layer_scale"), None)

    condition_dim, condition_bins = _parse_conditioners(raw)

    return ModelSpec(
        lm_weights_name=str(_field(raw, "moshi_name")),
        mimi_weights_name=str(_field(raw, "mimi_name")),
        tokenizer_name=str(_field(raw, "tokenizer_name")),
        dim=int(raw["dim"]),
        num_layers=int(raw["num_layers"]),
        num_heads=int(raw["num_heads"]),
        context=int(raw["context"]),
        max_period=int(raw["max_period"]),
        hidden_scale=float(raw["hidden_scale"]),
        text_card=int(raw["text_card"]),
        text_padding_id=int(raw["existing_text_padding_id"]),
        n_q=int(n_q),
        dep_q=int(dep_q),
        card=int(raw["card"]),
        delays=delays,
        depformer_dim=int(raw["depformer_dim"]),
        depformer_num_layers=int(raw["depformer_num_layers"]),
        depformer_num_heads=int(raw["depformer_num_heads"]),
        depformer_dim_feedforward=int(raw["depformer_dim_feedforward"]),
        depformer_context=int(raw["depformer_context"]),
        condition_dim=condition_dim,
        condition_bins=condition_bins,
    )


def _parse_conditioners(raw: dict[str, Any]) -> tuple[int, int]:
    """Validate the `description` lookup-table conditioner and return its shape."""
    conditioners = _mapping(_field(raw, "conditioners"), "conditioners")
    if set(conditioners) != {"description"}:
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} declares conditioners {sorted(conditioners)}, but "
            f"this implementation requires exactly one named 'description'."
        )
    description = _mapping(conditioners["description"], "conditioners.description")
    _require("conditioners.description.type", description.get("type"), "lut")

    lut = _mapping(description.get("lut"), "conditioners.description.lut")
    labels = tuple(lut.get("possible_values", ()))
    if labels != CONDITION_LABELS:
        raise IncompatibleArtifactError(
            f"{CONFIG_FILENAME} conditioner 'description' offers labels "
            f"{list(labels)}, but this implementation requires "
            f"{list(CONDITION_LABELS)}. Voice conditioning and any later "
            f"classifier-free guidance depend on this exact ordering."
        )

    # The conditioner embedding shapes are derived from these two numbers, so
    # leaving them unchecked would let drift stay self-consistent and pass.
    _require("conditioners.description.lut.dim", _field(lut, "dim"), 16)
    _require("conditioners.description.lut.n_bins", _field(lut, "n_bins"), 31)

    fuser = _mapping(_field(raw, "fuser"), "fuser")
    _require("fuser.sum", fuser.get("sum"), ["description"])

    return int(lut["dim"]), int(lut["n_bins"])
