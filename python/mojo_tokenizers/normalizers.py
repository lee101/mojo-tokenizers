"""Normalizer subset used by common tokenizer training pipelines."""

from __future__ import annotations

import unicodedata


class Lowercase:
    def normalize_str(self, sequence: str) -> str:
        return sequence.lower()


class NFC:
    def normalize_str(self, sequence: str) -> str:
        return unicodedata.normalize("NFC", sequence)


class NFD:
    def normalize_str(self, sequence: str) -> str:
        return unicodedata.normalize("NFD", sequence)


class NFKC:
    def normalize_str(self, sequence: str) -> str:
        return unicodedata.normalize("NFKC", sequence)


class NFKD:
    def normalize_str(self, sequence: str) -> str:
        return unicodedata.normalize("NFKD", sequence)


class Sequence:
    def __init__(self, normalizers):
        self.normalizers = list(normalizers)

    def normalize_str(self, sequence: str) -> str:
        for normalizer in self.normalizers:
            sequence = normalizer.normalize_str(sequence)
        return sequence

