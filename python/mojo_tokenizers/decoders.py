"""Decoder subset."""

from __future__ import annotations


class WordPiece:
    def __init__(self, prefix: str = "##", cleanup: bool = True):
        self.prefix = prefix
        self.cleanup = bool(cleanup)

    def decode(self, tokens: list[str]) -> str:
        text = ""
        for token in tokens:
            if token.startswith(self.prefix):
                text += token[len(self.prefix) :]
            elif text:
                text += " " + token
            else:
                text = token
        if self.cleanup:
            for before, after in (
                (" .", "."),
                (" ,", ","),
                (" !", "!"),
                (" ?", "?"),
                (" ' ", "'"),
                (" n't", "n't"),
            ):
                text = text.replace(before, after)
        return text


class Fuse:
    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)

