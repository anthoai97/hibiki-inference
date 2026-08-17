"""Tests for deriving the Hibiki model contract from the bundle configuration."""

from __future__ import annotations

import unittest

from hibiki_mlx.models.lm import LmConfig


def released_config(**overrides: object) -> dict:
    """The architecture fields of the released ``config.json``."""
    config = {
        "dim": 2048,
        "text_card": 48000,
        "existing_text_padding_id": 3,
        "n_q": 16,
        "dep_q": 8,
        "card": 2048,
        "num_heads": 16,
        "num_layers": 16,
        "hidden_scale": 4.125,
        "causal": True,
        "layer_scale": None,
        "context": 500,
        "max_period": 100000,
        "gating": "silu",
        "norm": "rms_norm_f32",
        "positional_embedding": "rope",
        "depformer_dim": 1024,
        "depformer_dim_feedforward": 4224,
        "depformer_num_heads": 16,
        "depformer_num_layers": 6,
        "depformer_causal": True,
        "depformer_layer_scale": None,
        "depformer_context": 16,
        "depformer_max_period": 10000,
        "depformer_pos_emb": "none",
        "depformer_weights_per_step": True,
        "delays": [0] + ([0] + [2] * 7) * 2,
        "conditioners": {
            "description": {
                "type": "lut",
                "lut": {
                    "n_bins": 31,
                    "dim": 16,
                    "tokenizer": "noop",
                    "possible_values": ["very_bad", "bad", "neutral", "good", "very_good"],
                },
            }
        },
    }
    config.update(overrides)
    return config


class LmConfigTests(unittest.TestCase):
    def test_splits_the_sixteen_audio_streams_into_target_and_source(self) -> None:
        config = LmConfig.from_config_dict(released_config())

        self.assertEqual(config.audio_codebooks, 16)
        self.assertEqual(config.target_codebooks, 8)
        self.assertEqual(config.source_codebooks, 8)

    def test_drops_the_text_delay_from_the_audio_delays(self) -> None:
        config = LmConfig.from_config_dict(released_config())

        self.assertEqual(config.audio_delays, ([0] + [2] * 7) * 2)

    def test_derives_the_gated_branch_width_from_the_hidden_scale(self) -> None:
        config = LmConfig.from_config_dict(released_config())

        # 4.125 * 2048 = 8448, and a gated branch is two thirds of that.
        self.assertEqual(config.transformer.dim_feedforward, 8448)
        self.assertEqual(2 * config.transformer.dim_feedforward // 3, 5632)
        self.assertEqual(2 * config.depformer.transformer.dim_feedforward // 3, 2816)

    def test_reads_the_no_text_token_from_the_bundle(self) -> None:
        config = LmConfig.from_config_dict(released_config())

        self.assertEqual(config.text_padding_token, 3)

    def test_follows_the_bundle_when_it_moves_the_no_text_token(self) -> None:
        config = LmConfig.from_config_dict(released_config(existing_text_padding_id=7))

        self.assertEqual(config.text_padding_token, 7)

    def test_keeps_the_extra_input_only_text_token(self) -> None:
        config = LmConfig.from_config_dict(released_config())

        self.assertEqual(config.text_in_vocab_size, 48001)
        self.assertEqual(config.text_out_vocab_size, 48000)
        self.assertEqual(config.audio_vocab_size, 2049)
        self.assertEqual(config.audio_padding_token, 2048)

    def test_rejects_a_delay_vector_that_does_not_cover_every_stream(self) -> None:
        with self.assertRaises(ValueError):
            LmConfig.from_config_dict(released_config(delays=[0, 0, 2]))

    def test_rejects_shared_depth_transformer_weights(self) -> None:
        with self.assertRaises(ValueError):
            LmConfig.from_config_dict(released_config(depformer_weights_per_step=False))

    def test_rejects_an_unsupported_gating(self) -> None:
        with self.assertRaises(ValueError):
            LmConfig.from_config_dict(released_config(gating="gelu"))
