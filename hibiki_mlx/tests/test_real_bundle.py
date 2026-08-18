"""Model-backed tests that load the released artifact bundle.

The bundle is never downloaded here. Point ``HIBIKI_MODEL_DIR`` at a local
artifact directory to run these; without it they skip, as the shared testing
contract in ``docs/smoke-test.md`` allows for local runs.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mlx.core as mx

from hibiki_mlx.audio import read_pcm
from hibiki_mlx.inference import load_model
from hibiki_mlx.sampling import Sampler
from hibiki_mlx.session import InferenceSession

MODEL_DIRECTORY = os.environ.get("HIBIKI_MODEL_DIR")
ASSETS = Path(__file__).resolve().parents[2] / "assets"
SHORT_FORM_SOURCE = ASSETS / "short-form/source/cvss-fr2en-test-idx14410-20011543.wav"

# Greedy decoding removes sampling from the comparison, so this is what the
# reference `moshi_mlx.run_inference` produces for the same input and bundle.
# It is a parity fixture: a change here means the model path changed.
EXPECTED_GREEDY_TRANSLATION = (
    "His sister, Marie Therese Gergis, is a producer of fiction and documentaries."
)


@unittest.skipUnless(MODEL_DIRECTORY, "HIBIKI_MODEL_DIR is not set")
class LoadModelTests(unittest.TestCase):
    model = None

    @classmethod
    def setUpClass(cls) -> None:
        assert MODEL_DIRECTORY is not None
        cls.model = load_model(artifact_directory=Path(MODEL_DIRECTORY))

    def test_loads_the_generator_in_the_released_precision(self) -> None:
        model = self.model
        self.assertEqual(model.lm.text_linear.weight.dtype, mx.bfloat16)
        self.assertEqual(model.lm.text_linear.weight.shape, (48000, 2048))
        self.assertEqual(len(model.lm.transformer.layers), 16)
        self.assertEqual(len(model.lm.depformer.slices), 8)
        self.assertEqual(len(model.lm.audio_embs), 16)

    def test_keeps_the_codec_at_the_precision_it_ships_as(self) -> None:
        model = self.model
        codebook = model.mimi.quantizer.rvq_first.vq.layers[0].codebook
        self.assertEqual(codebook.embedding_sum.dtype, mx.float32)
        self.assertEqual(model.mimi.cfg.quantizer_nq, 8)

    def test_drops_only_the_codebooks_beyond_the_eight_hibiki_uses(self) -> None:
        self.assertEqual(len(self.model.unused_codec_weights), 24 * 3)

    def test_derives_the_codebook_centroids_after_loading(self) -> None:
        # Centroids are derived from the loaded weights by a step that is easy
        # to skip. If it were skipped every centroid would be zero, every
        # distance would tie, and every codebook would answer 0 forever.
        model = self.model
        cache = model.mimi.make_encoder_cache()
        model.mimi.reset_state()
        frames = [
            model.mimi.encode_step(
                mx.random.normal((1, 1, model.mimi.cfg.frame_size)) * 0.1, cache
            )
            for _ in range(3)
        ]

        codes = mx.concatenate(frames, axis=-1)

        self.assertGreater(int(codes.max()), 0)
        self.assertGreater(len({int(code) for code in codes.reshape(-1).tolist()}), 1)

    def test_encodes_a_frame_of_silence_into_eight_codes(self) -> None:
        model = self.model
        cache = model.mimi.make_encoder_cache()
        model.mimi.reset_state()

        codes = model.mimi.encode_step(mx.zeros((1, 1, model.mimi.cfg.frame_size)), cache)

        self.assertEqual(codes.shape, (1, 8, 1))
        self.assertLess(int(codes.max()), 2048)

    def test_translates_the_short_form_asset_as_the_reference_does(self) -> None:
        if not SHORT_FORM_SOURCE.is_file():
            self.skipTest(f"{SHORT_FORM_SOURCE} is not checked out")
        session = InferenceSession(
            self.model,
            text_sampler=Sampler(temp=0),
            audio_sampler=Sampler(temp=0),
        )

        results = session.push_pcm(read_pcm(SHORT_FORM_SOURCE))
        results += session.finish()

        self.assertEqual(session.text.strip(), EXPECTED_GREEDY_TRANSLATION)
        # Every frame but the first two carries a complete target audio frame.
        audio_frames = [result for result in results if result.pcm is not None]
        self.assertEqual(len(audio_frames), len(results) - 2)
        self.assertEqual(audio_frames[0].audio_frame_index, 0)
        self.assertEqual(audio_frames[0].pcm.shape, (self.model.mimi.cfg.frame_size,))

    def test_reports_text_and_audio_on_their_own_frame_indices(self) -> None:
        if not SHORT_FORM_SOURCE.is_file():
            self.skipTest(f"{SHORT_FORM_SOURCE} is not checked out")
        session = InferenceSession(
            self.model,
            text_sampler=Sampler(temp=0),
            audio_sampler=Sampler(temp=0),
        )

        # The model clock comes from the codec, not from a literal.
        self.assertAlmostEqual(session.seconds_per_frame, 0.08)

        frames = 5
        results = session.push_pcm(read_pcm(SHORT_FORM_SOURCE)[: frames * session.frame_size])

        self.assertEqual([result.text_frame_index for result in results], [0, 1, 2, 3, 4])
        self.assertEqual(
            [result.audio_frame_index for result in results], [None, None, 0, 1, 2]
        )
        self.assertAlmostEqual(results[4].text_time, 0.32)
        self.assertAlmostEqual(results[4].audio_time, 0.16)

    def test_predicts_the_no_text_token_for_the_first_step(self) -> None:
        model = self.model
        cache = model.lm.make_transformer_cache()
        start_token = mx.array([[model.lm_config.text_out_vocab_size]])

        state, logits = model.lm.forward_text(start_token, cache)

        self.assertEqual(state.shape, (1, 1, 2048))
        self.assertEqual(logits.shape, (1, 1, 48000))
        # The bundle's own no-text id is what a run that has heard nothing yet
        # should predict.
        self.assertEqual(int(logits.argmax()), model.lm_config.text_padding_token)
