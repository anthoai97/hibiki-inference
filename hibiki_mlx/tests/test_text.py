"""Tests for incremental decoding of the text stream."""

from __future__ import annotations

import unittest

from hibiki_mlx.text import TextDecoder


class FakeTokenizer:
    """Renders ids as words, joining them the way SentencePiece would.

    Crucially, a piece's rendering depends on its neighbours: the id for a
    suffix attaches to the previous word rather than starting a new one.
    """

    WORDS = {3: "three", 10: "Her", 11: "sister", 12: "is", 13: "here"}
    SUFFIXES = {20: "'s"}

    def decode(self, tokens: list[int]) -> str:
        text = ""
        for token in tokens:
            if token in self.SUFFIXES:
                text += self.SUFFIXES[token]
            else:
                text += (" " if text else "") + self.WORDS[token]
        return text


class TextDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = TextDecoder(FakeTokenizer(), no_text_token=3)

    def test_emits_each_token_as_it_arrives(self) -> None:
        self.assertEqual(self.decoder.push(10), "Her")
        self.assertEqual(self.decoder.push(11), " sister")
        self.assertEqual(self.decoder.push(12), " is")

    def test_attaches_a_suffix_to_the_word_before_it(self) -> None:
        self.decoder.push(11)

        self.assertEqual(self.decoder.push(20), "'s")

    def test_never_sends_the_control_tokens_to_the_tokenizer(self) -> None:
        self.assertIsNone(self.decoder.push(3))
        self.assertIsNone(self.decoder.push(0))
        self.assertEqual(self.decoder.push(10), "Her")

    def test_takes_the_no_text_token_from_the_bundle_that_declares_it(self) -> None:
        decoder = TextDecoder(FakeTokenizer(), no_text_token=12)

        # 12 is this bundle's no-text token, so it is never decoded...
        self.assertIsNone(decoder.push(12))
        # ...and 3 is an ordinary word again.
        self.assertEqual(decoder.push(3), "three")

    def test_accumulates_the_whole_translation(self) -> None:
        for token in (10, 11, 3, 12, 0, 13):
            self.decoder.push(token)

        self.assertEqual(self.decoder.text, "Her sister is here")

    def test_reset_starts_a_new_translation(self) -> None:
        self.decoder.push(10)
        self.decoder.reset()

        self.assertEqual(self.decoder.text, "")
        self.assertEqual(self.decoder.push(11), "sister")
