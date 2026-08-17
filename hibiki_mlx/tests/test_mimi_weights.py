"""Tests for converting the released codec weights into local parameters."""

from __future__ import annotations

import unittest

import mlx.core as mx

from hibiki_mlx.models.mimi import MimiWeightError, remap_released_weights


def remap(names: dict[str, tuple[int, ...]], codebooks: int = 8):
    raw = {name: mx.zeros(shape) for name, shape in names.items()}
    return remap_released_weights(raw, codebooks=codebooks)


class RemapNameTests(unittest.TestCase):
    def test_maps_the_seanet_sequence_positions_onto_named_layers(self) -> None:
        weights, _ = remap(
            {
                "encoder.model.0.conv.conv.bias": (64,),
                "encoder.model.1.block.1.conv.conv.bias": (32,),
                "encoder.model.1.block.3.conv.conv.bias": (64,),
                "encoder.model.3.conv.conv.bias": (128,),
                "encoder.model.14.conv.conv.bias": (512,),
                "decoder.model.0.conv.conv.bias": (1024,),
                "decoder.model.2.convtr.convtr.bias": (512,),
                "decoder.model.3.block.1.conv.conv.bias": (256,),
                "decoder.model.14.conv.conv.bias": (1,),
            }
        )

        self.assertEqual(
            sorted(weights),
            [
                "decoder.final_conv1d.conv.conv.bias",
                "decoder.init_conv1d.conv.conv.bias",
                "decoder.layers.0.residuals.0.block.0.conv.conv.bias",
                "decoder.layers.0.upsample.convtr.convtr.bias",
                "encoder.final_conv1d.conv.conv.bias",
                "encoder.init_conv1d.conv.conv.bias",
                "encoder.layers.0.downsample.conv.conv.bias",
                "encoder.layers.0.residuals.0.block.0.conv.conv.bias",
                "encoder.layers.0.residuals.0.block.1.conv.conv.bias",
            ],
        )

    def test_strips_the_private_prefixes_the_release_uses(self) -> None:
        weights, _ = remap({"quantizer.rvq_first.vq.layers.0._codebook._initialized": (1,)})

        self.assertEqual(list(weights), ["quantizer.rvq_first.vq.layers.0.codebook.initialized"])

    def test_moves_the_attention_and_feed_forward_names_under_their_modules(self) -> None:
        weights, _ = remap(
            {
                "encoder_transformer.transformer.layers.0.self_attn.in_proj_weight": (1536, 512),
                "encoder_transformer.transformer.layers.0.linear1.weight": (2048, 512),
                "encoder_transformer.transformer.layers.0.linear2.weight": (512, 2048),
            }
        )

        self.assertEqual(
            sorted(weights),
            [
                "encoder_transformer.transformer.layers.0.gating.linear1.weight",
                "encoder_transformer.transformer.layers.0.gating.linear2.weight",
                "encoder_transformer.transformer.layers.0.self_attn.in_proj.weight",
            ],
        )

    def test_rejects_a_sequence_position_that_holds_no_layer(self) -> None:
        with self.assertRaises(MimiWeightError):
            remap({"encoder.model.13.conv.conv.bias": (64,)})

    def test_rejects_an_unexpected_position_inside_a_residual_block(self) -> None:
        with self.assertRaises(MimiWeightError):
            remap({"encoder.model.1.block.2.conv.conv.bias": (32,)})


class RemapLayoutTests(unittest.TestCase):
    def test_transposes_convolution_weights_into_the_mlx_layout(self) -> None:
        weights, _ = remap({"encoder.model.0.conv.conv.weight": (64, 1, 7)})

        self.assertEqual(weights["encoder.init_conv1d.conv.conv.weight"].shape, (64, 7, 1))

    def test_transposes_transposed_convolution_weights(self) -> None:
        weights, _ = remap({"upsample.convtr.convtr.convtr.weight": (512, 1, 4)})

        self.assertEqual(weights["upsample.convtr.convtr.convtr.weight"].shape, (1, 4, 512))

    def test_transposes_the_quantizer_projections(self) -> None:
        weights, _ = remap({"quantizer.rvq_first.input_proj.weight": (256, 512, 1)})

        self.assertEqual(weights["quantizer.rvq_first.input_proj.weight"].shape, (256, 1, 512))


class UnusedCodebookTests(unittest.TestCase):
    def test_drops_only_the_codebooks_beyond_the_ones_hibiki_uses(self) -> None:
        names = {
            f"quantizer.rvq_rest.vq.layers.{index}._codebook.cluster_usage": (2048,)
            for index in range(31)
        }
        names["quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"] = (2048,)

        weights, dropped = remap(names, codebooks=8)

        self.assertEqual(len(weights), 8)
        self.assertEqual(len(dropped), 24)
        self.assertTrue(
            all(name.startswith("quantizer.rvq_rest.vq.layers.") for name in dropped),
            dropped,
        )
