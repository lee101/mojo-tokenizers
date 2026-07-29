"""Small, compatible pre-tokenizer subset."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\w+|[^\w\s]+", re.UNICODE)
_WHITESPACE_SPLIT = re.compile(r"\S+", re.UNICODE)


class Whitespace:
    def pre_tokenize_offsets(self, sequence: str) -> list[tuple[int, int]]:
        return [match.span() for match in _WHITESPACE.finditer(sequence)]

    def pre_tokenize_str(self, sequence: str) -> list[tuple[str, tuple[int, int]]]:
        return [
            (sequence[start:end], (start, end))
            for start, end in self.pre_tokenize_offsets(sequence)
        ]


class WhitespaceSplit:
    def pre_tokenize_offsets(self, sequence: str) -> list[tuple[int, int]]:
        return [match.span() for match in _WHITESPACE_SPLIT.finditer(sequence)]

    def pre_tokenize_str(self, sequence: str) -> list[tuple[str, tuple[int, int]]]:
        return [
            (sequence[start:end], (start, end))
            for start, end in self.pre_tokenize_offsets(sequence)
        ]
