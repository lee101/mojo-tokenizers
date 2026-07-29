"""Trainers for the three supported subword models."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np

from ._lib import addr, i64, lib, power_of_two
from .models import BPE, Unigram, WordPiece, _merged_token


def _special_text(token) -> str:
    return token.content if hasattr(token, "content") else str(token)


class Trainer:
    def train(self, model, words: Counter[str]) -> None:
        raise NotImplementedError


class BpeTrainer(Trainer):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        show_progress: bool = True,
        progress_format: str = "indicatif",
        special_tokens: list[str] | None = None,
        limit_alphabet: int | None = None,
        initial_alphabet: list[str] | None = None,
        continuing_subword_prefix: str | None = None,
        end_of_word_suffix: str | None = None,
        max_token_length: int | None = None,
        words: dict[str, int] | None = None,
    ):
        self.vocab_size = int(vocab_size)
        self.min_frequency = int(min_frequency)
        self.show_progress = bool(show_progress)
        self.progress_format = progress_format
        self.special_tokens = list(special_tokens or [])
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = list(initial_alphabet or [])
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.max_token_length = max_token_length
        self.words = dict(words or {})

    def train(self, model: BPE, words: Counter[str]) -> None:
        words = Counter(words)
        words.update(self.words)
        prefix = (
            self.continuing_subword_prefix
            if self.continuing_subword_prefix is not None
            else model.continuing_subword_prefix
        )
        suffix = (
            self.end_of_word_suffix
            if self.end_of_word_suffix is not None
            else model.end_of_word_suffix
        )
        alphabet = _alphabet(words, self.initial_alphabet, self.limit_alphabet)
        tokens = [_special_text(token) for token in self.special_tokens]
        symbols: set[str] = set()
        for word in words:
            for index, character in enumerate(word):
                if character not in alphabet:
                    continue
                symbol = (prefix or "") + character if index else character
                if index == len(word) - 1 and suffix:
                    symbol += suffix
                symbols.add(symbol)
        if not prefix and not suffix:
            symbols.update(alphabet)
        tokens.extend(sorted(symbols))
        tokens = list(dict.fromkeys(tokens))
        vocab = {token: index for index, token in enumerate(tokens)}
        sequences: list[list[int]] = []
        frequencies: list[int] = []
        for word, frequency in sorted(words.items()):
            sequence: list[int] = []
            for index, character in enumerate(word):
                if character not in alphabet:
                    continue
                symbol = (prefix or "") + character if index else character
                if index == len(word) - 1 and suffix:
                    symbol += suffix
                if symbol in vocab:
                    sequence.append(vocab[symbol])
            if sequence:
                sequences.append(sequence)
                frequencies.append(int(frequency))
        merges: list[tuple[str, str]] = []
        while len(vocab) < self.vocab_size:
            token_counts, pairs = _counts(sequences, frequencies, len(vocab))
            del token_counts
            candidates = [
                (count, left, right)
                for (left, right), count in pairs.items()
                if count >= self.min_frequency
            ]
            if not candidates:
                break
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            selected = None
            for _, left, right in candidates:
                merged = _merged_token(tokens[left], tokens[right], prefix)
                if self.max_token_length is not None and len(merged) > self.max_token_length:
                    continue
                selected = (left, right, merged)
                break
            if selected is None:
                break
            left, right, merged = selected
            if merged in vocab:
                result = vocab[merged]
            else:
                result = len(tokens)
                vocab[merged] = result
                tokens.append(merged)
            merges.append((tokens[left], tokens[right]))
            _merge_sequences(sequences, left, right, result)
            if result < len(tokens) - 1:
                break
        model.vocab = vocab
        model.merges = merges
        model.continuing_subword_prefix = prefix
        model.end_of_word_suffix = suffix
        model._refresh()


class WordPieceTrainer(Trainer):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        show_progress: bool = True,
        special_tokens: list[str] | None = None,
        limit_alphabet: int | None = None,
        initial_alphabet: list[str] | None = None,
        continuing_subword_prefix: str = "##",
        end_of_word_suffix: str | None = None,
    ):
        self.vocab_size = int(vocab_size)
        self.min_frequency = int(min_frequency)
        self.show_progress = bool(show_progress)
        self.special_tokens = list(special_tokens or [])
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = list(initial_alphabet or [])
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix

    def train(self, model: WordPiece, words: Counter[str]) -> None:
        alphabet = _alphabet(words, self.initial_alphabet, self.limit_alphabet)
        prefix = self.continuing_subword_prefix
        tokens = [_special_text(token) for token in self.special_tokens]
        tokens.extend(sorted(alphabet))
        continuation = []
        seen_continuation: set[str] = set()
        for word in words:
            for character in word[1:]:
                token = prefix + character
                if character in alphabet and token not in seen_continuation:
                    continuation.append(token)
                    seen_continuation.add(token)
        tokens.extend(continuation)
        tokens = list(dict.fromkeys(tokens))
        vocab = {token: index for index, token in enumerate(tokens)}
        sequences: list[list[int]] = []
        frequencies: list[int] = []
        for word, frequency in sorted(words.items()):
            sequence = []
            for index, character in enumerate(word):
                if character not in alphabet:
                    continue
                token = character if index == 0 else prefix + character
                sequence.append(vocab[token])
            if sequence:
                sequences.append(sequence)
                frequencies.append(int(frequency))
        while len(vocab) < self.vocab_size:
            _, pairs = _counts(sequences, frequencies, len(vocab))
            candidates = []
            for (left, right), count in pairs.items():
                if count < self.min_frequency:
                    continue
                candidates.append((count, left, right))
            if not candidates:
                break
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            _, left, right = candidates[0]
            merged = _merged_token(tokens[left], tokens[right], prefix)
            if merged in vocab:
                result = vocab[merged]
            else:
                result = len(tokens)
                vocab[merged] = result
                tokens.append(merged)
            _merge_sequences(sequences, left, right, result)
            if result < len(tokens) - 1:
                break
        model.vocab = vocab
        model.continuing_subword_prefix = prefix
        model._refresh()


class UnigramTrainer(Trainer):
    def __init__(
        self,
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: list[str] | None = None,
        initial_alphabet: list[str] | None = None,
        shrinking_factor: float = 0.75,
        unk_token: str | None = None,
        max_piece_length: int = 16,
        n_sub_iterations: int = 2,
    ):
        self.vocab_size = int(vocab_size)
        self.show_progress = bool(show_progress)
        self.special_tokens = list(special_tokens or [])
        self.initial_alphabet = list(initial_alphabet or [])
        self.shrinking_factor = float(shrinking_factor)
        self.unk_token = unk_token
        self.max_piece_length = int(max_piece_length)
        self.n_sub_iterations = int(n_sub_iterations)
        if not 0.0 < self.shrinking_factor < 1.0:
            raise ValueError("shrinking_factor must be between 0 and 1")

    def train(self, model: Unigram, words: Counter[str]) -> None:
        substring_counts: Counter[str] = Counter()
        required = set(character for word in words for character in word)
        required.update(item[0] for item in self.initial_alphabet if item)
        for word, frequency in words.items():
            for start in range(len(word)):
                limit = min(len(word), start + self.max_piece_length)
                for end in range(start + 1, limit + 1):
                    substring_counts[word[start:end]] += frequency
        seed_limit = max(self.vocab_size * 8, self.vocab_size + len(required))
        candidates = {
            piece
            for piece, _ in substring_counts.most_common(seed_limit)
        }
        candidates.update(required)
        total = sum(substring_counts[piece] for piece in candidates) or 1
        scores = {
            piece: math.log(max(substring_counts[piece], 1) / total)
            for piece in candidates
        }
        target = max(0, self.vocab_size - len(self.special_tokens))
        expected: dict[str, float] = {}
        while True:
            for _ in range(max(1, self.n_sub_iterations)):
                expected = _unigram_expectation(words, scores, self.max_piece_length)
                expected_total = sum(expected.values()) or 1.0
                scores = {
                    piece: math.log(max(expected.get(piece, 1e-12), 1e-12) / expected_total)
                    for piece in scores
                }
            if len(scores) <= target:
                break
            next_size = max(target, int(len(scores) * self.shrinking_factor))
            removable = sorted(
                (piece for piece in scores if piece not in required),
                key=lambda piece: (expected.get(piece, 0.0), len(piece), piece),
            )
            remove_count = min(len(scores) - next_size, len(removable))
            if not remove_count:
                break
            for piece in removable[:remove_count]:
                del scores[piece]
        specials = [_special_text(token) for token in self.special_tokens]
        if self.unk_token and self.unk_token not in specials:
            specials.insert(0, self.unk_token)
        ordered = sorted(scores, key=lambda piece: (-scores[piece], piece))
        vocab: list[tuple[str, float]] = []
        seen: set[str] = set()
        for special in specials:
            if special not in seen:
                vocab.append((special, 0.0))
                seen.add(special)
        for piece in ordered:
            if piece not in seen and len(vocab) < self.vocab_size:
                vocab.append((piece, scores[piece]))
                seen.add(piece)
        model._vocab_scores = vocab
        model.unk_id = (
            next(
                (index for index, (token, _) in enumerate(vocab) if token == self.unk_token),
                None,
            )
            if self.unk_token is not None
            else None
        )
        model._refresh()


def _alphabet(
    words: Counter[str],
    initial_alphabet: list[str],
    limit: int | None,
) -> set[str]:
    counts: Counter[str] = Counter()
    for word, frequency in words.items():
        for character in word:
            counts[character] += frequency
    initial = {item[0] for item in initial_alphabet if item}
    if limit is None:
        return set(counts) | initial
    ranked = sorted(counts, key=lambda character: (-counts[character], character))
    return set(ranked[: max(0, int(limit))]) | initial


def _counts(
    sequences: list[list[int]], frequencies: list[int], vocab_size: int
) -> tuple[list[int], dict[tuple[int, int], int]]:
    if len(sequences) != len(frequencies):
        raise ValueError("sequences and frequencies must have the same length")
    if vocab_size < 0:
        raise ValueError("vocab_size must be non-negative")
    max_i64 = np.iinfo(np.int64).max
    totals = [0] * vocab_size
    for sequence, frequency in zip(sequences, frequencies):
        if (
            not isinstance(frequency, (int, np.integer))
            or isinstance(frequency, (bool, np.bool_))
            or frequency < 0
            or frequency > max_i64
        ):
            raise ValueError("frequencies must be non-negative int64 values")
        for token_id in sequence:
            if (
                not isinstance(token_id, (int, np.integer))
                or isinstance(token_id, (bool, np.bool_))
                or token_id < 0
                or token_id >= vocab_size
            ):
                raise ValueError("token IDs must be integers within the vocabulary")
            totals[int(token_id)] += int(frequency)
            if totals[int(token_id)] > max_i64:
                raise OverflowError("weighted token count exceeds int64")
    total_ids = sum(map(len, sequences))
    if total_ids < 4_096:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for sequence, frequency in zip(sequences, frequencies):
            for left, right in zip(sequence, sequence[1:]):
                pair_counts[left, right] += frequency
                if pair_counts[left, right] > max_i64:
                    raise OverflowError("weighted pair count exceeds int64")
        return totals, dict(pair_counts)

    offsets = [0]
    flat: list[int] = []
    for sequence in sequences:
        flat.extend(sequence)
        offsets.append(len(flat))
    adjacent = sum(max(0, len(sequence) - 1) for sequence in sequences)
    cap = power_of_two(max(4, adjacent * 2))
    ids = i64(flat)
    offsets_array = i64(offsets)
    frequencies_array = i64(frequencies)
    token_counts = i64(size=vocab_size)
    left = i64(size=cap)
    right = i64(size=cap)
    counts = i64(size=cap)
    lib().mt_count_tokens_pairs(
        addr(ids, np.int64),
        addr(offsets_array, np.int64),
        addr(frequencies_array, np.int64),
        len(sequences),
        addr(token_counts, np.int64),
        vocab_size,
        addr(left, np.int64),
        addr(right, np.int64),
        addr(counts, np.int64),
        cap,
    )
    pairs = {
        (int(left[index]), int(right[index])): int(counts[index])
        for index in range(cap)
        if left[index] >= 0
    }
    return token_counts[:vocab_size].tolist(), pairs


def _merge_sequences(
    sequences: list[list[int]], left: int, right: int, result: int
) -> None:
    for sequence in sequences:
        read = 0
        write = 0
        length = len(sequence)
        while read < length:
            if (
                read + 1 < length
                and sequence[read] == left
                and sequence[read + 1] == right
            ):
                sequence[write] = result
                read += 2
            else:
                sequence[write] = sequence[read]
                read += 1
            write += 1
        del sequence[write:]


def _logadd(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def _unigram_expectation(
    words: Counter[str], scores: dict[str, float], max_piece_length: int
) -> dict[str, float]:
    expected = {piece: 0.0 for piece in scores}
    for word, frequency in words.items():
        length = len(word)
        alpha = [-math.inf] * (length + 1)
        alpha[0] = 0.0
        edges: list[list[tuple[int, str]]] = [[] for _ in range(length)]
        for start in range(length):
            for end in range(start + 1, min(length, start + max_piece_length) + 1):
                piece = word[start:end]
                if piece in scores:
                    edges[start].append((end, piece))
                    alpha[end] = _logadd(alpha[end], alpha[start] + scores[piece])
        partition = alpha[length]
        if partition == -math.inf:
            continue
        beta = [-math.inf] * (length + 1)
        beta[length] = 0.0
        for start in range(length - 1, -1, -1):
            for end, piece in edges[start]:
                beta[start] = _logadd(beta[start], scores[piece] + beta[end])
        for start, outgoing in enumerate(edges):
            for end, piece in outgoing:
                posterior = math.exp(
                    alpha[start] + scores[piece] + beta[end] - partition
                )
                expected[piece] += frequency * posterior
    return expected
