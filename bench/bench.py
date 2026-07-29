"""End-to-end benchmarks against Hugging Face tokenizers."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_tokenizers as mt  # noqa: E402
from mojo_tokenizers import models as mm  # noqa: E402
from mojo_tokenizers import pre_tokenizers as mp  # noqa: E402
from mojo_tokenizers import trainers as mtr  # noqa: E402
from tokenizers import Tokenizer as RustTokenizer  # noqa: E402
from tokenizers import models as rm  # noqa: E402
from tokenizers import pre_tokenizers as rp  # noqa: E402
from tokenizers import trainers as rtr  # noqa: E402


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def train_pair(model_kind: str, corpus: list[str], vocab_size: int):
    if model_kind == "BPE":
        ours = mt.Tokenizer(mm.BPE(unk_token="<unk>"))
        theirs = RustTokenizer(rm.BPE(unk_token="<unk>"))
        ours_trainer = mtr.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            show_progress=False,
        )
        theirs_trainer = rtr.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<unk>"],
            show_progress=False,
        )
    elif model_kind == "WordPiece":
        ours = mt.Tokenizer(mm.WordPiece(unk_token="[UNK]"))
        theirs = RustTokenizer(rm.WordPiece(unk_token="[UNK]"))
        ours_trainer = mtr.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]"],
            show_progress=False,
        )
        theirs_trainer = rtr.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]"],
            show_progress=False,
        )
    else:
        ours = mt.Tokenizer(mm.Unigram())
        theirs = RustTokenizer(rm.Unigram())
        ours_trainer = mtr.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
        )
        theirs_trainer = rtr.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
        )
    ours.pre_tokenizer = mp.Whitespace()
    theirs.pre_tokenizer = rp.Whitespace()
    ours.train_from_iterator(corpus, ours_trainer)
    theirs.train_from_iterator(corpus, theirs_trainer)
    return ours, theirs, ours_trainer, theirs_trainer


def main() -> None:
    base = [
        "low lower lowest newer wider",
        "token tokenizer tokenized training trainer",
        "mojo modular model merge word piece unigram",
        "encoding offsets vocabulary unknown character",
    ]
    corpus = base * 500
    bpe_ours, bpe_theirs, _, _ = train_pair("BPE", corpus, 160)
    wp_ours, wp_theirs, _, _ = train_pair("WordPiece", corpus, 160)

    unigram_vocab = [
        ("<unk>", 0.0),
        ("token", -0.2),
        ("izer", -0.5),
        ("training", -0.4),
        ("mojo", -0.4),
        ("model", -0.4),
        ("word", -0.5),
        ("piece", -0.5),
    ]
    alphabet = sorted(set(" ".join(base)))
    unigram_vocab.extend((character, -4.0) for character in alphabet)
    ug_ours = mt.Tokenizer(mm.Unigram(unigram_vocab, unk_id=0))
    ug_theirs = RustTokenizer(rm.Unigram(unigram_vocab, unk_id=0))
    ug_ours.pre_tokenizer = mp.Whitespace()
    ug_theirs.pre_tokenizer = rp.Whitespace()

    text = (" ".join(base) + " ") * 2_000
    assert bpe_ours.encode(text).tokens == bpe_theirs.encode(text).tokens
    assert wp_ours.encode(text).tokens == wp_theirs.encode(text).tokens
    assert ug_ours.encode(text).tokens == ug_theirs.encode(text).tokens

    cases = [
        (
            f"BPE encode ({len(text) / 1e6:.2f}M chars)",
            lambda: bpe_ours.encode(text),
            lambda: bpe_theirs.encode(text),
            5,
        ),
        (
            f"WordPiece encode ({len(text) / 1e6:.2f}M chars)",
            lambda: wp_ours.encode(text),
            lambda: wp_theirs.encode(text),
            5,
        ),
        (
            f"Unigram encode ({len(text) / 1e6:.2f}M chars)",
            lambda: ug_ours.encode(text),
            lambda: ug_theirs.encode(text),
            5,
        ),
    ]

    training_corpus = base * 1_000

    def train_only(kind: str, mojo: bool):
        if kind == "BPE":
            tokenizer = (
                mt.Tokenizer(mm.BPE(unk_token="<unk>"))
                if mojo
                else RustTokenizer(rm.BPE(unk_token="<unk>"))
            )
            trainer = (
                mtr.BpeTrainer(
                    vocab_size=160,
                    min_frequency=2,
                    special_tokens=["<unk>"],
                    show_progress=False,
                )
                if mojo
                else rtr.BpeTrainer(
                    vocab_size=160,
                    min_frequency=2,
                    special_tokens=["<unk>"],
                    show_progress=False,
                )
            )
        elif kind == "WordPiece":
            tokenizer = (
                mt.Tokenizer(mm.WordPiece())
                if mojo
                else RustTokenizer(rm.WordPiece())
            )
            trainer = (
                mtr.WordPieceTrainer(
                    vocab_size=160,
                    min_frequency=2,
                    special_tokens=["[UNK]"],
                    show_progress=False,
                )
                if mojo
                else rtr.WordPieceTrainer(
                    vocab_size=160,
                    min_frequency=2,
                    special_tokens=["[UNK]"],
                    show_progress=False,
                )
            )
        else:
            tokenizer = (
                mt.Tokenizer(mm.Unigram())
                if mojo
                else RustTokenizer(rm.Unigram())
            )
            trainer = (
                mtr.UnigramTrainer(
                    vocab_size=160,
                    special_tokens=["<unk>"],
                    unk_token="<unk>",
                    show_progress=False,
                )
                if mojo
                else rtr.UnigramTrainer(
                    vocab_size=160,
                    special_tokens=["<unk>"],
                    unk_token="<unk>",
                    show_progress=False,
                )
            )
        tokenizer.pre_tokenizer = mp.Whitespace() if mojo else rp.Whitespace()
        tokenizer.train_from_iterator(training_corpus, trainer)

    for kind in ("BPE", "WordPiece", "Unigram"):
        cases.append(
            (
                f"{kind} train (4k lines, vocab 160)",
                lambda kind=kind: train_only(kind, True),
                lambda kind=kind: train_only(kind, False),
                3,
            )
        )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-tokenizers | tokenizers 0.23.1 | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs, repeat in cases:
        ours()
        theirs()
        ours_time = best_time(ours, repeat)
        theirs_time = best_time(theirs, repeat)
        print(
            f"| {name} | {ours_time * 1e3:.2f} ms | "
            f"{theirs_time * 1e3:.2f} ms | {theirs_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
