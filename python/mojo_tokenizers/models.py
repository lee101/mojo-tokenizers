"""BPE, WordPiece, and Unigram models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Iterable

import numpy as np

from ._lib import addr, checked_count, f64, i64, lib, power_of_two, u32


@dataclass(frozen=True)
class Token:
    id: int
    value: str
    offsets: tuple[int, int]


def _fnv(codepoints: Iterable[int]) -> int:
    value = 1469598103934665603
    for codepoint in codepoints:
        value = ((value ^ int(codepoint)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


class _VocabIndex:
    def __init__(self, vocab: dict[str, int]):
        _validate_vocab(vocab)
        max_id = max(vocab.values(), default=-1)
        chunks: list[np.ndarray] = []
        offsets = [0] * (max_id + 1)
        lengths = [-1] * (max_id + 1)
        cursor = 0
        encoded: list[tuple[int, np.ndarray]] = []
        for token, token_id in vocab.items():
            chars = np.frombuffer(token.encode("utf-32-le"), dtype=np.uint32)
            encoded.append((token_id, chars))
            offsets[token_id] = cursor
            lengths[token_id] = len(chars)
            chunks.append(chars)
            cursor += len(chars)
        self.chars = (
            np.ascontiguousarray(np.concatenate(chunks), dtype=np.uint32)
            if cursor
            else np.zeros(1, dtype=np.uint32)
        )
        self.offsets = i64(offsets)
        self.lengths = i64(lengths)
        self.cap = power_of_two(max(4, len(vocab) * 2))
        self.table = i64(size=self.cap, fill=-1)
        for token_id, chars in encoded:
            slot = _fnv(chars) & (self.cap - 1)
            while self.table[slot] != -1:
                slot = (slot + 1) & (self.cap - 1)
            self.table[slot] = token_id


class Model:
    type = "Model"

    def tokenize(self, sequence: str) -> list[Token]:
        ids, tokens, starts, ends = self._encode(sequence, [(0, len(sequence))])
        return [
            Token(int(token_id), token, (int(start), int(end)))
            for token_id, token, start, end in zip(ids, tokens, starts, ends)
        ]

    def token_to_id(self, token: str) -> int | None:
        return self.vocab.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._id_to_token.get(int(token_id))

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


class BPE(Model):
    type = "BPE"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        cache_capacity: int | None = None,
        dropout: float | None = None,
        unk_token: str | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        fuse_unk: bool | None = None,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ):
        if dropout is not None and not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if dropout not in (None, 0.0):
            raise NotImplementedError("BPE dropout is not supported")
        self.vocab = dict(vocab or {})
        self.merges = [tuple(pair) for pair in (merges or [])]
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = bool(fuse_unk)
        self.byte_fallback = bool(byte_fallback)
        self.ignore_merges = bool(ignore_merges)
        self._refresh()

    def _refresh(self) -> None:
        _validate_vocab(self.vocab)
        self._id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        self._unk_id = self.vocab.get(self.unk_token, -1)
        self._cache: dict[
            str, tuple[list[int], list[str], list[int], list[int]]
        ] = {}
        self._cache_limit = (
            10_000 if self.cache_capacity is None else max(0, int(self.cache_capacity))
        )
        cap = power_of_two(max(4, len(self.merges) * 2))
        self._pair_cap = cap
        self._pair_left = i64(size=cap, fill=-1)
        self._pair_right = i64(size=cap, fill=-1)
        self._pair_rank = i64(size=cap, fill=-1)
        self._pair_result = i64(size=cap, fill=-1)
        valid_merges: list[tuple[str, str]] = []
        for rank, (left_token, right_token) in enumerate(self.merges):
            left = self.vocab.get(left_token)
            right = self.vocab.get(right_token)
            merged_token = _merged_token(
                left_token, right_token, self.continuing_subword_prefix
            )
            result = self.vocab.get(merged_token)
            if left is None or right is None or result is None:
                raise ValueError(
                    f"merge {(left_token, right_token)!r} references a token "
                    "missing from the vocabulary"
                )
            slot = _pair_hash(left, right) & (cap - 1)
            while self._pair_left[slot] != -1:
                slot = (slot + 1) & (cap - 1)
            self._pair_left[slot] = left
            self._pair_right[slot] = right
            self._pair_rank[slot] = rank
            self._pair_result[slot] = result
            valid_merges.append((left_token, right_token))
        self.merges = valid_merges

    def _initial_symbol(self, character: str, index: int, length: int) -> str:
        token = character
        if index and self.continuing_subword_prefix:
            token = self.continuing_subword_prefix + token
        if index == length - 1 and self.end_of_word_suffix:
            token += self.end_of_word_suffix
        return token

    def _encode(
        self, text: str, segments: list[tuple[int, int]]
    ) -> tuple[list[int], list[str], list[int], list[int]]:
        if not self.vocab:
            raise ValueError("BPE vocabulary is empty; train the model first")
        self._last_segment_counts = None
        if not segments:
            return [], [], [], []
        if self._cache_limit == 0 or len(segments) == 1:
            return self._encode_uncached(text, segments)

        pieces = [text[start:end] for start, end in segments]
        missing = list(dict.fromkeys(piece for piece in pieces if piece not in self._cache))
        if len(missing) > 128:
            return self._encode_uncached(text, segments)
        for piece in missing:
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[piece] = self._encode_uncached(piece, [(0, len(piece))])

        ids: list[int] = []
        tokens: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        if not self.fuse_unk:
            cached = [self._cache[piece] for piece in pieces]
            self._last_segment_counts = [len(encoded[0]) for encoded in cached]
            ids = [token_id for encoded in cached for token_id in encoded[0]]
            tokens = [token for encoded in cached for token in encoded[1]]
            starts = [
                segment_start + start
                for (segment_start, _), encoded in zip(segments, cached)
                for start in encoded[2]
            ]
            ends = [
                segment_start + end
                for (segment_start, _), encoded in zip(segments, cached)
                for end in encoded[3]
            ]
            return ids, tokens, starts, ends
        for (segment_start, _), piece in zip(segments, pieces):
            piece_ids, piece_tokens, piece_starts, piece_ends = self._cache[piece]
            for token_id, token, start, end in zip(
                piece_ids, piece_tokens, piece_starts, piece_ends
            ):
                absolute_start = segment_start + start
                absolute_end = segment_start + end
                if (
                    self.fuse_unk
                    and token_id == self._unk_id
                    and ids
                    and ids[-1] == self._unk_id
                    and ends[-1] == absolute_start
                ):
                    ends[-1] = absolute_end
                else:
                    ids.append(token_id)
                    tokens.append(token)
                    starts.append(absolute_start)
                    ends.append(absolute_end)
        return ids, tokens, starts, ends

    def _encode_uncached(
        self, text: str, segments: list[tuple[int, int]]
    ) -> tuple[list[int], list[str], list[int], list[int]]:
        self._last_segment_counts = None
        input_ids: list[int] = []
        input_starts: list[int] = []
        input_ends: list[int] = []
        segment_offsets = [0]
        for start, end in segments:
            piece = text[start:end]
            if self.ignore_merges and piece in self.vocab:
                input_ids.append(self.vocab[piece])
                input_starts.append(start)
                input_ends.append(end)
                segment_offsets.append(len(input_ids))
                continue
            for index, character in enumerate(piece):
                token = self._initial_symbol(character, index, len(piece))
                token_id = self.vocab.get(token)
                absolute = start + index
                if token_id is not None:
                    input_ids.append(token_id)
                    input_starts.append(absolute)
                    input_ends.append(absolute + 1)
                    continue
                if self.byte_fallback:
                    byte_tokens = [f"<0x{byte:02X}>" for byte in character.encode()]
                    if all(byte_token in self.vocab for byte_token in byte_tokens):
                        for byte_token in byte_tokens:
                            input_ids.append(self.vocab[byte_token])
                            input_starts.append(absolute)
                            input_ends.append(absolute + 1)
                        continue
                if self._unk_id < 0:
                    continue
                input_ids.append(self._unk_id)
                input_starts.append(absolute)
                input_ends.append(absolute + 1)
            segment_offsets.append(len(input_ids))
        if not input_ids:
            return [], [], [], []
        ids_array = i64(input_ids)
        starts_array = i64(input_starts)
        ends_array = i64(input_ends)
        offsets_array = i64(segment_offsets)
        capacity = len(input_ids)
        result_ids = i64(size=capacity)
        result_starts = i64(size=capacity)
        result_ends = i64(size=capacity)
        work_ids = i64(size=capacity)
        work_starts = i64(size=capacity)
        work_ends = i64(size=capacity)
        count = lib().mt_bpe_encode(
            addr(ids_array, np.int64),
            addr(starts_array, np.int64),
            addr(ends_array, np.int64),
            addr(offsets_array, np.int64),
            len(segments),
            addr(self._pair_left, np.int64),
            addr(self._pair_right, np.int64),
            addr(self._pair_rank, np.int64),
            addr(self._pair_result, np.int64),
            self._pair_cap,
            self._unk_id,
            int(self.fuse_unk),
            addr(result_ids, np.int64),
            addr(result_starts, np.int64),
            addr(result_ends, np.int64),
            addr(work_ids, np.int64),
            addr(work_starts, np.int64),
            addr(work_ends, np.int64),
        )
        count = checked_count(count, capacity, "BPE encoder")
        ids = result_ids[:count].tolist()
        starts = result_starts[:count].tolist()
        ends = result_ends[:count].tolist()
        tokens = [self._id_to_token[token_id] for token_id in ids]
        return ids, tokens, starts, ends

    @classmethod
    def from_file(
        cls,
        vocab: str,
        merge: str,
        **kwargs,
    ) -> "BPE":
        with open(vocab, encoding="utf-8") as handle:
            vocabulary = json.load(handle)
        merges = _read_merges(merge)
        return cls(vocabulary, merges, **kwargs)

    def save(self, folder: str, prefix: str | None = None) -> list[str]:
        os.makedirs(folder, exist_ok=True)
        stem = f"{prefix}-" if prefix else ""
        vocab_path = os.path.join(folder, stem + "vocab.json")
        merges_path = os.path.join(folder, stem + "merges.txt")
        with open(vocab_path, "w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, ensure_ascii=False)
        with open(merges_path, "w", encoding="utf-8") as handle:
            handle.write("#version: 0.2\n")
            for left, right in self.merges:
                handle.write(f"{left} {right}\n")
        return [vocab_path, merges_path]


class WordPiece(Model):
    type = "WordPiece"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
        continuing_subword_prefix: str = "##",
    ):
        self.vocab = dict(vocab or {})
        self.unk_token = unk_token
        self.max_input_chars_per_word = int(max_input_chars_per_word)
        self.continuing_subword_prefix = continuing_subword_prefix
        self._refresh()

    def _refresh(self) -> None:
        _validate_vocab(self.vocab)
        self._id_to_token = {token_id: token for token, token_id in self.vocab.items()}
        self._unk_id = self.vocab.get(self.unk_token, -1)
        self._index = _VocabIndex(self.vocab)
        self._prefix = u32(self.continuing_subword_prefix)
        self._prefix_len = len(self.continuing_subword_prefix)

    def _encode(
        self, text: str, segments: list[tuple[int, int]]
    ) -> tuple[list[int], list[str], list[int], list[int]]:
        if self._unk_id < 0:
            raise ValueError(f"WordPiece unknown token {self.unk_token!r} is not in vocab")
        if not segments:
            return [], [], [], []
        chars = u32(text)
        starts_in = i64([start for start, _ in segments])
        ends_in = i64([end for _, end in segments])
        capacity = max(1, len(text) + len(segments))
        result_ids = i64(size=capacity)
        result_starts = i64(size=capacity)
        result_ends = i64(size=capacity)
        count = lib().mt_wordpiece_encode(
            addr(chars, np.uint32),
            addr(starts_in, np.int64),
            addr(ends_in, np.int64),
            len(segments),
            addr(self._prefix, np.uint32),
            self._prefix_len,
            addr(self._index.chars, np.uint32),
            addr(self._index.offsets, np.int64),
            addr(self._index.lengths, np.int64),
            addr(self._index.table, np.int64),
            self._index.cap,
            self._unk_id,
            self.max_input_chars_per_word,
            addr(result_ids, np.int64),
            addr(result_starts, np.int64),
            addr(result_ends, np.int64),
        )
        count = checked_count(count, capacity, "WordPiece encoder")
        ids = result_ids[:count].tolist()
        starts = result_starts[:count].tolist()
        ends = result_ends[:count].tolist()
        return ids, [self._id_to_token[token_id] for token_id in ids], starts, ends

    @classmethod
    def from_file(cls, vocab: str, **kwargs) -> "WordPiece":
        with open(vocab, encoding="utf-8") as handle:
            vocabulary = {line.rstrip("\n"): index for index, line in enumerate(handle)}
        return cls(vocabulary, **kwargs)

    def save(self, folder: str, prefix: str | None = None) -> list[str]:
        os.makedirs(folder, exist_ok=True)
        stem = f"{prefix}-" if prefix else ""
        path = os.path.join(folder, stem + "vocab.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for index in range(max(self._id_to_token, default=-1) + 1):
                handle.write(self._id_to_token[index] + "\n")
        return [path]


class Unigram(Model):
    type = "Unigram"

    def __init__(
        self,
        vocab: list[tuple[str, float]] | None = None,
        unk_id: int | None = None,
        byte_fallback: bool | None = None,
        alpha: float | None = None,
        nbest_size: int | None = None,
    ):
        if byte_fallback:
            raise NotImplementedError("Unigram byte fallback is not supported")
        if alpha is not None or nbest_size is not None:
            raise NotImplementedError("Unigram sampling is not supported")
        self._vocab_scores = [(token, float(score)) for token, score in (vocab or [])]
        if any(not math.isfinite(score) for _, score in self._vocab_scores):
            raise ValueError("Unigram scores must be finite")
        self.unk_id = unk_id
        self.byte_fallback = bool(byte_fallback)
        self.alpha = alpha
        self.nbest_size = nbest_size
        self._refresh()

    def _refresh(self) -> None:
        self.vocab = {token: index for index, (token, _) in enumerate(self._vocab_scores)}
        self._id_to_token = {
            index: token for index, (token, _) in enumerate(self._vocab_scores)
        }
        tokens = [token for token, _ in self._vocab_scores]
        self._scores = f64([score for _, score in self._vocab_scores])
        self._index = _VocabIndex(self.vocab)
        self._max_piece_chars = max((len(token) for token in tokens), default=1)
        self._cache: dict[
            str, tuple[list[int], list[str], list[int], list[int]]
        ] = {}

    def _encode(
        self, text: str, segments: list[tuple[int, int]]
    ) -> tuple[list[int], list[str], list[int], list[int]]:
        self._last_segment_counts = None
        if self.unk_id is None and any(
            character not in self.vocab
            for start, end in segments
            for character in text[start:end]
        ):
            raise ValueError("Unigram cannot encode unknown characters without unk_id")
        if not segments:
            return [], [], [], []
        if len(segments) == 1:
            return self._encode_uncached(text, segments)

        pieces = [text[start:end] for start, end in segments]
        missing = list(dict.fromkeys(piece for piece in pieces if piece not in self._cache))
        if len(missing) > 128:
            return self._encode_uncached(text, segments)
        for piece in missing:
            if len(self._cache) >= 10_000:
                self._cache.pop(next(iter(self._cache)))
            self._cache[piece] = self._encode_uncached(piece, [(0, len(piece))])

        unknown = int(self.unk_id if self.unk_id is not None else -1)
        ids: list[int] = []
        tokens: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        cached = [self._cache[piece] for piece in pieces]
        if not any(unknown in encoded[0] for encoded in cached):
            self._last_segment_counts = [len(encoded[0]) for encoded in cached]
            ids = [token_id for encoded in cached for token_id in encoded[0]]
            tokens = [token for encoded in cached for token in encoded[1]]
            starts = [
                segment_start + start
                for (segment_start, _), encoded in zip(segments, cached)
                for start in encoded[2]
            ]
            ends = [
                segment_start + end
                for (segment_start, _), encoded in zip(segments, cached)
                for end in encoded[3]
            ]
            return ids, tokens, starts, ends
        for (segment_start, _), encoded in zip(segments, cached):
            piece_ids, piece_tokens, piece_starts, piece_ends = encoded
            if unknown not in piece_ids:
                ids.extend(piece_ids)
                tokens.extend(piece_tokens)
                starts.extend(segment_start + start for start in piece_starts)
                ends.extend(segment_start + end for end in piece_ends)
                continue
            for token_id, token, start, end in zip(
                piece_ids, piece_tokens, piece_starts, piece_ends
            ):
                absolute_start = segment_start + start
                absolute_end = segment_start + end
                if (
                    token_id == unknown
                    and ids
                    and ids[-1] == unknown
                    and ends[-1] == absolute_start
                ):
                    ends[-1] = absolute_end
                    tokens[-1] = text[starts[-1] : absolute_end]
                else:
                    ids.append(token_id)
                    tokens.append(token)
                    starts.append(absolute_start)
                    ends.append(absolute_end)
        return ids, tokens, starts, ends

    def _encode_uncached(
        self, text: str, segments: list[tuple[int, int]]
    ) -> tuple[list[int], list[str], list[int], list[int]]:
        self._last_segment_counts = None
        unknown = int(self.unk_id if self.unk_id is not None else -1)
        chars = u32(text)
        starts_in = i64([start for start, _ in segments])
        ends_in = i64([end for _, end in segments])
        capacity = max(1, len(text) + len(segments))
        dp = f64(size=len(text) + 1)
        previous = i64(size=len(text) + 1)
        previous_id = i64(size=len(text) + 1)
        temp_ids = i64(size=capacity)
        temp_starts = i64(size=capacity)
        temp_ends = i64(size=capacity)
        result_ids = i64(size=capacity)
        result_starts = i64(size=capacity)
        result_ends = i64(size=capacity)
        min_score = min((score for _, score in self._vocab_scores), default=-10.0)
        count = lib().mt_unigram_encode(
            addr(chars, np.uint32),
            addr(starts_in, np.int64),
            addr(ends_in, np.int64),
            len(segments),
            addr(self._index.chars, np.uint32),
            addr(self._index.offsets, np.int64),
            addr(self._index.lengths, np.int64),
            addr(self._scores, np.float64),
            addr(self._index.table, np.int64),
            self._index.cap,
            unknown,
            min_score - 10.0,
            self._max_piece_chars,
            addr(dp, np.float64),
            addr(previous, np.int64),
            addr(previous_id, np.int64),
            addr(temp_ids, np.int64),
            addr(temp_starts, np.int64),
            addr(temp_ends, np.int64),
            addr(result_ids, np.int64),
            addr(result_starts, np.int64),
            addr(result_ends, np.int64),
        )
        count = checked_count(count, capacity, "Unigram encoder")
        ids = result_ids[:count].tolist()
        starts = result_starts[:count].tolist()
        ends = result_ends[:count].tolist()
        tokens = [
            text[start:end] if token_id == unknown else self._id_to_token[token_id]
            for token_id, start, end in zip(ids, starts, ends)
        ]
        return ids, tokens, starts, ends

    def save(self, folder: str, prefix: str | None = None) -> list[str]:
        os.makedirs(folder, exist_ok=True)
        stem = f"{prefix}-" if prefix else ""
        path = os.path.join(folder, stem + "unigram.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"vocab": self._vocab_scores, "unk_id": self.unk_id},
                handle,
                ensure_ascii=False,
            )
        return [path]


def _pair_hash(left: int, right: int) -> int:
    return (
        (left * 11400714819323198485) ^ (right * 14029467366897019727)
    ) & 0xFFFFFFFFFFFFFFFF


def _validate_vocab(vocab: dict[str, int]) -> None:
    ids = list(vocab.values())
    i64(ids)
    if any(token_id < 0 for token_id in ids):
        raise ValueError("vocabulary IDs must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError("vocabulary IDs must be unique")


def _merged_token(left: str, right: str, prefix: str | None) -> str:
    if prefix and right.startswith(prefix):
        right = right[len(prefix) :]
    return left + right


def _read_merges(path: str) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right = line.split()
            merges.append((left, right))
    return merges
