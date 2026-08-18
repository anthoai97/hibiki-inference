"""Incremental decoding of the English text stream.

SentencePiece pieces do not decode independently: a piece's rendering depends on
what precedes it, and the word boundary marker belongs to the sequence rather
than to any one piece. Decoding the accumulated ids and emitting the newly added
suffix therefore gives the right text, where mapping each id to its piece and
substituting the boundary marker does not.
"""

from __future__ import annotations

from typing import Protocol

# The end-of-padding control id. Unlike the no-text id, the artifact bundle does
# not name this one, so it is fixed here.
END_OF_PADDING_TOKEN = 0


class Tokenizer(Protocol):
    """The whole of what text decoding needs from a tokenizer.

    ``tokens`` is positional-only because the real implementer, SentencePiece,
    names that parameter something else.
    """

    def decode(self, tokens: list[int], /) -> str: ...


class TextDecoder:
    """Turns a stream of sampled text tokens into a stream of text fragments.

    ``no_text_token`` has no default: every bundle names its own, and guessing
    would silently emit a control id as text.
    """

    def __init__(self, tokenizer: Tokenizer, *, no_text_token: int):
        self._tokenizer = tokenizer
        self._control_tokens = frozenset({END_OF_PADDING_TOKEN, no_text_token})
        self._tokens: list[int] = []
        self._decoded = ""

    def reset(self) -> None:
        self._tokens = []
        self._decoded = ""

    @property
    def text(self) -> str:
        """Everything decoded so far in this run."""
        return self._decoded

    def push(self, token: int) -> str | None:
        """Add one sampled token, returning the text it completes, if any."""
        if token in self._control_tokens:
            return None
        self._tokens.append(token)
        decoded = self._tokenizer.decode(self._tokens)
        fragment = decoded[len(self._decoded) :]
        self._decoded = decoded
        return fragment or None
