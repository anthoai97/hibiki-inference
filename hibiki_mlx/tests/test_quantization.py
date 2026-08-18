"""Tests for the compatible Q8/Q4 Hibiki LM artifact contract."""

from __future__ import annotations

import unittest

import mlx.nn as nn

from hibiki_mlx.artifacts import (
    QUANTIZATION_CONFIG_KEY,
    QuantizationSpec,
    quantize_linear_layers,
)


class QuantizationSpecTests(unittest.TestCase):
    def test_uses_kyutais_q8_and_q4_group_sizes(self) -> None:
        self.assertEqual(QuantizationSpec.for_bits(8).group_size, 64)
        self.assertEqual(QuantizationSpec.for_bits(4).group_size, 32)

    def test_round_trips_the_bundle_config_contract(self) -> None:
        original = QuantizationSpec.for_bits(4)

        restored = QuantizationSpec.from_config({QUANTIZATION_CONFIG_KEY: original.as_config()})

        self.assertEqual(restored, original)

    def test_rejects_an_unsupported_group_size(self) -> None:
        with self.assertRaises(ValueError):
            QuantizationSpec.from_config(
                {QUANTIZATION_CONFIG_KEY: {"bits": 4, "group_size": 64, "format": "linear-v1"}}
            )


class LinearSelectionTests(unittest.TestCase):
    def test_quantizes_only_linear_layers_with_compatible_shapes(self) -> None:
        model = nn.Sequential(nn.Linear(64, 64), nn.Linear(16, 64))

        quantize_linear_layers(model, QuantizationSpec.for_bits(4))

        self.assertIsInstance(model.layers[0], nn.QuantizedLinear)
        self.assertIsInstance(model.layers[1], nn.Linear)
