"""Tokenizer pipeline and Encoding container."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from . import decoders, normalizers, pre_tokenizers
from .models import BPE, Model, Unigram, WordPiece
from .trainers import BpeTrainer, Trainer, UnigramTrainer, WordPieceTrainer


@dataclass(frozen=True)
class AddedToken:
    content: str
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True
    special: bool = False

    def __str__(self) -> str:
        return self.content


class Encoding:
    def __init__(
        self,
        ids: list[int] | None = None,
        tokens: list[str] | None = None,
        offsets: list[tuple[int, int]] | None = None,
        word_ids: list[int | None] | None = None,
        type_ids: list[int] | None = None,
        sequence_ids: list[int | None] | None = None,
        special_tokens_mask: list[int] | None = None,
        _take_ownership: bool = False,
    ):
        if _take_ownership:
            self.ids = ids or []
            self.tokens = tokens or []
            self.offsets = offsets or []
            self.word_ids = word_ids or [None] * len(self.ids)
            self.type_ids = type_ids or [0] * len(self.ids)
            self.sequence_ids = sequence_ids or [0] * len(self.ids)
            self.special_tokens_mask = special_tokens_mask or [0] * len(self.ids)
        else:
            self.ids = list(ids or [])
            self.tokens = list(tokens or [])
            self.offsets = list(offsets or [])
            self.word_ids = list(word_ids or [None] * len(self.ids))
            self.type_ids = list(type_ids or [0] * len(self.ids))
            self.sequence_ids = list(sequence_ids or [0] * len(self.ids))
            self.special_tokens_mask = list(special_tokens_mask or [0] * len(self.ids))
        self.attention_mask = [1] * len(self.ids)
        self.overflowing: list[Encoding] = []

    def __len__(self) -> int:
        return len(self.ids)

    def token_to_chars(self, token_index: int) -> tuple[int, int] | None:
        if token_index < 0 or token_index >= len(self.offsets):
            return None
        return self.offsets[token_index]

    def char_to_token(
        self, char_pos: int, sequence_index: int = 0
    ) -> int | None:
        for index, ((start, end), sequence) in enumerate(
            zip(self.offsets, self.sequence_ids)
        ):
            if sequence == sequence_index and start <= char_pos < end:
                return index
        return None

    def token_to_word(self, token_index: int) -> int | None:
        return self.word_ids[token_index] if 0 <= token_index < len(self.word_ids) else None

    def word_to_tokens(
        self, word_index: int, sequence_index: int = 0
    ) -> tuple[int, int] | None:
        positions = [
            index
            for index, (word, sequence) in enumerate(
                zip(self.word_ids, self.sequence_ids)
            )
            if word == word_index and sequence == sequence_index
        ]
        return (positions[0], positions[-1] + 1) if positions else None

    def word_to_chars(
        self, word_index: int, sequence_index: int = 0
    ) -> tuple[int, int] | None:
        token_range = self.word_to_tokens(word_index, sequence_index)
        if token_range is None:
            return None
        start, end = token_range
        return self.offsets[start][0], self.offsets[end - 1][1]


class Tokenizer:
    def __init__(self, model: Model):
        self.model = model
        self.normalizer = None
        self.pre_tokenizer = None
        self.decoder = None
        self.post_processor = None
        self._special_tokens: set[str] = set()

    def _normalize(self, sequence: str) -> str:
        normalized = (
            self.normalizer.normalize_str(sequence)
            if self.normalizer is not None
            else sequence
        )
        if len(normalized) != len(sequence):
            raise NotImplementedError(
                "normalizers that change character count are not supported"
            )
        return normalized

    def _segments(
        self, sequence: str, is_pretokenized: bool
    ) -> tuple[str, list[tuple[int, int]], list[int]]:
        if is_pretokenized:
            if not isinstance(sequence, (list, tuple)):
                raise TypeError("pretokenized input must be a list of strings")
            pieces = [self._normalize(str(piece)) for piece in sequence]
            joined = " ".join(pieces)
            segments: list[tuple[int, int]] = []
            cursor = 0
            for piece in pieces:
                segments.append((cursor, cursor + len(piece)))
                cursor += len(piece) + 1
            return joined, segments, list(range(len(segments)))
        if not isinstance(sequence, str):
            raise TypeError("sequence must be a string")
        normalized = self._normalize(sequence)
        if self.pre_tokenizer is None:
            segments = [(0, len(normalized))] if normalized else []
        elif hasattr(self.pre_tokenizer, "pre_tokenize_offsets"):
            segments = self.pre_tokenizer.pre_tokenize_offsets(normalized)
        else:
            segments = [
                offsets
                for _, offsets in self.pre_tokenizer.pre_tokenize_str(normalized)
            ]
        return normalized, segments, list(range(len(segments)))

    def _encode_one(
        self, sequence, is_pretokenized: bool, sequence_id: int
    ) -> Encoding:
        text, segments, segment_words = self._segments(sequence, is_pretokenized)
        ids, tokens, starts, ends = self.model._encode(text, segments)
        segment_counts = getattr(self.model, "_last_segment_counts", None)
        if segment_counts is not None and sum(segment_counts) == len(ids):
            words = [
                segment_words[index]
                for index, count in enumerate(segment_counts)
                for _ in range(count)
            ]
        else:
            words: list[int | None] = []
            segment_index = 0
            for start, end in zip(starts, ends):
                while (
                    segment_index < len(segments)
                    and start >= segments[segment_index][1]
                ):
                    segment_index += 1
                word = None
                if segment_index < len(segments):
                    seg_start, seg_end = segments[segment_index]
                    if seg_start <= start and end <= seg_end:
                        word = segment_words[segment_index]
                words.append(word)
        if is_pretokenized:
            for index, word in enumerate(words):
                if word is not None:
                    base = segments[word][0]
                    starts[index] -= base
                    ends[index] -= base
        special_mask = [0] * len(tokens)
        return Encoding(
            ids,
            tokens,
            list(zip(starts, ends)),
            words,
            [sequence_id] * len(ids),
            [sequence_id] * len(ids),
            special_mask,
            _take_ownership=True,
        )

    def encode(
        self,
        sequence,
        pair=None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        del add_special_tokens
        first = self._encode_one(sequence, is_pretokenized, 0)
        if pair is None:
            return first
        second = self._encode_one(pair, is_pretokenized, 1)
        return Encoding(
            first.ids + second.ids,
            first.tokens + second.tokens,
            first.offsets + second.offsets,
            first.word_ids + second.word_ids,
            first.type_ids + second.type_ids,
            first.sequence_ids + second.sequence_ids,
            first.special_tokens_mask + second.special_tokens_mask,
        )

    def encode_batch(
        self,
        input,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        return [
            self.encode(
                item[0],
                item[1],
                is_pretokenized=is_pretokenized,
                add_special_tokens=add_special_tokens,
            )
            if isinstance(item, tuple)
            else self.encode(
                item,
                is_pretokenized=is_pretokenized,
                add_special_tokens=add_special_tokens,
            )
            for item in input
        ]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token = self.model.id_to_token(token_id)
            if token is None:
                continue
            if skip_special_tokens and token in self._special_tokens:
                continue
            tokens.append(token)
        if self.decoder is not None:
            return self.decoder.decode(tokens)
        return "".join(tokens)

    def decode_batch(
        self, sequences: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in sequences]

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        trainer: Trainer | None = None,
        length: int | None = None,
    ) -> None:
        del length
        if trainer is None:
            if isinstance(self.model, BPE):
                trainer = BpeTrainer()
            elif isinstance(self.model, WordPiece):
                trainer = WordPieceTrainer()
            elif isinstance(self.model, Unigram):
                trainer = UnigramTrainer()
            else:
                raise TypeError("no default trainer for this model")
        sequences: Counter[str] = Counter()
        for value in iterator:
            values = [value] if isinstance(value, str) else list(value)
            for sequence in values:
                sequences[str(sequence)] += 1
        words: Counter[str] = Counter()
        for sequence, frequency in sequences.items():
            text, segments, _ = self._segments(sequence, False)
            for start, end in segments:
                words[text[start:end]] += frequency
        trainer.train(self.model, words)
        self._special_tokens.update(
            token.content if hasattr(token, "content") else str(token)
            for token in getattr(trainer, "special_tokens", [])
        )

    def train(self, files: list[str], trainer: Trainer | None = None) -> None:
        def lines():
            for filename in files:
                with open(filename, encoding="utf-8") as handle:
                    yield from handle

        self.train_from_iterator(lines(), trainer)

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        del with_added_tokens
        return self.model.get_vocab()

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return len(self.get_vocab(with_added_tokens))

    def token_to_id(self, token: str) -> int | None:
        return self.model.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.model.id_to_token(token_id)

    def add_special_tokens(self, tokens: list[str | AddedToken]) -> int:
        before = len(self._special_tokens)
        self._special_tokens.update(
            token.content if isinstance(token, AddedToken) else str(token)
            for token in tokens
        )
        return len(self._special_tokens) - before

    def to_str(self, pretty: bool = False) -> str:
        payload = {
            "version": "1.0",
            "normalizer": _component_config(self.normalizer),
            "pre_tokenizer": _component_config(self.pre_tokenizer),
            "decoder": _component_config(self.decoder),
            "special_tokens": sorted(self._special_tokens),
            "model": _model_config(self.model),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)

    @classmethod
    def from_str(cls, json_string: str) -> "Tokenizer":
        payload = json.loads(json_string)
        model_config = payload["model"]
        model_type = model_config.pop("type")
        if model_type == "BPE":
            model = BPE(**model_config)
        elif model_type == "WordPiece":
            model = WordPiece(**model_config)
        elif model_type == "Unigram":
            model = Unigram(**model_config)
        else:
            raise ValueError(f"unsupported model type {model_type!r}")
        tokenizer = cls(model)
        tokenizer.normalizer = _load_component(payload.get("normalizer"), normalizers)
        tokenizer.pre_tokenizer = _load_component(
            payload.get("pre_tokenizer"), pre_tokenizers
        )
        tokenizer.decoder = _load_component(payload.get("decoder"), decoders)
        tokenizer._special_tokens = set(payload.get("special_tokens", []))
        return tokenizer

    @classmethod
    def from_file(cls, path: str) -> "Tokenizer":
        return cls.from_str(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty), encoding="utf-8")


def _component_config(component):
    if component is None:
        return None
    config = {"type": type(component).__name__}
    config.update(
        {
            key: value
            for key, value in vars(component).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    )
    return config


def _load_component(config, module):
    if not config:
        return None
    config = dict(config)
    cls = getattr(module, config.pop("type"))
    return cls(**config)


def _model_config(model: Model) -> dict:
    if isinstance(model, BPE):
        return {
            "type": "BPE",
            "vocab": model.vocab,
            "merges": model.merges,
            "cache_capacity": model.cache_capacity,
            "dropout": model.dropout,
            "unk_token": model.unk_token,
            "continuing_subword_prefix": model.continuing_subword_prefix,
            "end_of_word_suffix": model.end_of_word_suffix,
            "fuse_unk": model.fuse_unk,
            "byte_fallback": model.byte_fallback,
            "ignore_merges": model.ignore_merges,
        }
    if isinstance(model, WordPiece):
        return {
            "type": "WordPiece",
            "vocab": model.vocab,
            "unk_token": model.unk_token,
            "max_input_chars_per_word": model.max_input_chars_per_word,
            "continuing_subword_prefix": model.continuing_subword_prefix,
        }
    if isinstance(model, Unigram):
        return {
            "type": "Unigram",
            "vocab": model._vocab_scores,
            "unk_id": model.unk_id,
            "byte_fallback": model.byte_fallback,
            "alpha": model.alpha,
            "nbest_size": model.nbest_size,
        }
    raise TypeError(type(model).__name__)
