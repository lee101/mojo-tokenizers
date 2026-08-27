"""Behavioral parity with Hugging Face tokenizers 0.23.x."""

from __future__ import annotations

import json

import numpy as np
import pytest

import mojo_tokenizers as mt
from mojo_tokenizers import _lib as mojo_lib
from mojo_tokenizers import decoders as mojo_decoders
from mojo_tokenizers import models as mojo_models
from mojo_tokenizers import normalizers as mojo_normalizers
from mojo_tokenizers import pre_tokenizers as mojo_pre
from mojo_tokenizers import trainers as mojo_trainers

upstream = pytest.importorskip("tokenizers")
from tokenizers import decoders as upstream_decoders
from tokenizers import models as upstream_models
from tokenizers import normalizers as upstream_normalizers
from tokenizers import pre_tokenizers as upstream_pre
from tokenizers import trainers as upstream_trainers


def paired(model_ours, model_theirs, pre=None, normalizer=None):
    ours = mt.Tokenizer(model_ours)
    theirs = upstream.Tokenizer(model_theirs)
    if pre == "whitespace":
        ours.pre_tokenizer = mojo_pre.Whitespace()
        theirs.pre_tokenizer = upstream_pre.Whitespace()
    elif pre == "split":
        ours.pre_tokenizer = mojo_pre.WhitespaceSplit()
        theirs.pre_tokenizer = upstream_pre.WhitespaceSplit()
    if normalizer == "lower":
        ours.normalizer = mojo_normalizers.Lowercase()
        theirs.normalizer = upstream_normalizers.Lowercase()
    return ours, theirs


def assert_encoding_equal(ours, theirs):
    assert ours.ids == theirs.ids
    assert ours.tokens == theirs.tokens
    assert ours.offsets == theirs.offsets
    assert ours.word_ids == theirs.word_ids
    assert ours.type_ids == theirs.type_ids
    assert ours.sequence_ids == theirs.sequence_ids
    assert ours.attention_mask == theirs.attention_mask
    assert ours.special_tokens_mask == theirs.special_tokens_mask


def assert_token_behavior_equal(ours, theirs):
    assert ours.tokens == theirs.tokens
    assert ours.offsets == theirs.offsets
    assert ours.word_ids == theirs.word_ids
    assert ours.type_ids == theirs.type_ids
    assert ours.sequence_ids == theirs.sequence_ids
    assert ours.attention_mask == theirs.attention_mask
    assert ours.special_tokens_mask == theirs.special_tokens_mask


@pytest.fixture
def bpe_pair():
    vocab = {
        "<unk>": 0,
        "a": 1,
        "b": 2,
        "ab": 3,
        "é": 4,
        "aa": 5,
        "aaa": 6,
    }
    merges = [("a", "b"), ("a", "a"), ("aa", "a")]
    return paired(
        mojo_models.BPE(vocab, merges, unk_token="<unk>"),
        upstream_models.BPE(vocab, merges, unk_token="<unk>"),
    )


@pytest.mark.parametrize("text", ["ab", "aaaa", "ab éz", "", "éaba"])
def test_bpe_fixed_vocab_parity(bpe_pair, text):
    ours, theirs = bpe_pair
    assert_encoding_equal(ours.encode(text), theirs.encode(text))


def test_bpe_whitespace_pipeline_parity():
    vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3, "é": 4, "!": 5}
    merges = [("a", "b")]
    ours, theirs = paired(
        mojo_models.BPE(vocab, merges, unk_token="<unk>"),
        upstream_models.BPE(vocab, merges, unk_token="<unk>"),
        pre="whitespace",
    )
    assert_encoding_equal(ours.encode("ab éz!"), theirs.encode("ab éz!"))


def test_whitespace_split_pipeline_parity():
    vocab = {"[UNK]": 0, "hello": 1, "world!": 2}
    ours, theirs = paired(
        mojo_models.WordPiece(vocab),
        upstream_models.WordPiece(vocab),
        pre="split",
    )
    assert_encoding_equal(ours.encode("hello  world!"), theirs.encode("hello  world!"))


@pytest.mark.parametrize("fuse", [False, True])
def test_bpe_unknown_fusion_parity(fuse):
    vocab = {"?": 0, "a": 1}
    ours, theirs = paired(
        mojo_models.BPE(vocab, [], unk_token="?", fuse_unk=fuse),
        upstream_models.BPE(vocab, [], unk_token="?", fuse_unk=fuse),
    )
    assert_encoding_equal(ours.encode("xxax"), theirs.encode("xxax"))


@pytest.mark.parametrize("ignore_merges", [False, True])
def test_bpe_ignore_merges_parity(ignore_merges):
    vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3}
    ours, theirs = paired(
        mojo_models.BPE(
            vocab, [], unk_token="<unk>", ignore_merges=ignore_merges
        ),
        upstream_models.BPE(
            vocab, [], unk_token="<unk>", ignore_merges=ignore_merges
        ),
    )
    assert_encoding_equal(ours.encode("ab"), theirs.encode("ab"))


def test_bpe_prefix_suffix_parity():
    vocab = {"<unk>": 0, "a": 1, "##b</w>": 2, "ab</w>": 3}
    merges = [("a", "##b</w>")]
    kwargs = {
        "unk_token": "<unk>",
        "continuing_subword_prefix": "##",
        "end_of_word_suffix": "</w>",
    }
    ours, theirs = paired(
        mojo_models.BPE(vocab, merges, **kwargs),
        upstream_models.BPE(vocab, merges, **kwargs),
    )
    assert_encoding_equal(ours.encode("ab"), theirs.encode("ab"))


def test_bpe_byte_fallback_parity():
    vocab = {"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2}
    ours, theirs = paired(
        mojo_models.BPE(
            vocab, [], unk_token="<unk>", byte_fallback=True
        ),
        upstream_models.BPE(
            vocab, [], unk_token="<unk>", byte_fallback=True
        ),
    )
    assert_encoding_equal(ours.encode("é"), theirs.encode("é"))


@pytest.fixture
def wordpiece_pair():
    vocab = {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "w": 3,
        "##ide": 4,
        "##r": 5,
        "!": 6,
        "é": 7,
    }
    return paired(
        mojo_models.WordPiece(vocab),
        upstream_models.WordPiece(vocab),
        pre="whitespace",
    )


@pytest.mark.parametrize(
    "text",
    ["hello world!", "wider", "unknown", "é wider", ""],
)
def test_wordpiece_longest_match_parity(wordpiece_pair, text):
    ours, theirs = wordpiece_pair
    assert_encoding_equal(ours.encode(text), theirs.encode(text))


def test_wordpiece_repeated_segment_cache_parity(wordpiece_pair):
    ours, theirs = wordpiece_pair
    text = "hello wider unknown hello wider"
    assert_encoding_equal(ours.encode(text), theirs.encode(text))
    assert_encoding_equal(ours.encode(text), theirs.encode(text))


def test_wordpiece_max_chars_parity():
    vocab = {"[UNK]": 0, "a": 1, "##a": 2}
    ours, theirs = paired(
        mojo_models.WordPiece(vocab, max_input_chars_per_word=3),
        upstream_models.WordPiece(vocab, max_input_chars_per_word=3),
    )
    assert_encoding_equal(ours.encode("aaaa"), theirs.encode("aaaa"))


def test_wordpiece_sparse_vocab_ids_are_preserved():
    model = mojo_models.WordPiece({"[UNK]": 7, "a": 42, "##b": 99})
    assert model.tokenize("ab") == [
        mojo_models.Token(42, "a", (0, 1)),
        mojo_models.Token(99, "##b", (1, 2)),
    ]


@pytest.fixture
def unigram_pair():
    vocab = [
        ("<unk>", 0.0),
        ("a", -1.0),
        ("b", -2.0),
        ("ab", -1.5),
        ("aba", -1.6),
        ("é", -1.0),
    ]
    return paired(
        mojo_models.Unigram(vocab, unk_id=0),
        upstream_models.Unigram(vocab, unk_id=0),
    )


@pytest.mark.parametrize("text", ["ab", "ab éz", "ababa", "xyz", "éab", ""])
def test_unigram_viterbi_parity(unigram_pair, text):
    ours, theirs = unigram_pair
    assert_encoding_equal(ours.encode(text), theirs.encode(text))


def test_unigram_with_whitespace_parity():
    vocab = [("<unk>", 0.0), ("ab", -1.0), ("é", -1.0)]
    ours, theirs = paired(
        mojo_models.Unigram(vocab, unk_id=0),
        upstream_models.Unigram(vocab, unk_id=0),
        pre="whitespace",
    )
    assert_encoding_equal(ours.encode("ab éz"), theirs.encode("ab éz"))


def test_repeated_segment_caches_preserve_offsets_and_unknown_fusion():
    bpe_vocab = {"?": 0, "a": 1, "b": 2, "ab": 3}
    ours, theirs = paired(
        mojo_models.BPE(bpe_vocab, [("a", "b")], unk_token="?", fuse_unk=True),
        upstream_models.BPE(bpe_vocab, [("a", "b")], unk_token="?", fuse_unk=True),
        pre="whitespace",
    )
    text = "ab xx ab xx ab"
    assert_encoding_equal(ours.encode(text), theirs.encode(text))
    assert_encoding_equal(ours.encode(text), theirs.encode(text))

    unigram_vocab = [("<unk>", 0.0), ("ab", -0.5), ("a", -1.0), ("b", -1.0)]
    ours, theirs = paired(
        mojo_models.Unigram(unigram_vocab, unk_id=0),
        upstream_models.Unigram(unigram_vocab, unk_id=0),
        pre="whitespace",
    )
    assert_encoding_equal(ours.encode(text), theirs.encode(text))
    assert_encoding_equal(ours.encode(text), theirs.encode(text))


def test_training_repeated_lines_matches_expanded_corpus():
    repeated = mt.Tokenizer(mojo_models.BPE(unk_token="<unk>"))
    expanded = mt.Tokenizer(mojo_models.BPE(unk_token="<unk>"))
    repeated.pre_tokenizer = mojo_pre.Whitespace()
    expanded.pre_tokenizer = mojo_pre.Whitespace()
    trainer_args = {
        "vocab_size": 20,
        "special_tokens": ["<unk>"],
        "show_progress": False,
    }
    repeated.train_from_iterator(
        ["low lower", "low lower", "newer wider"],
        mojo_trainers.BpeTrainer(**trainer_args),
    )
    expanded.train_from_iterator(
        ["low lower", "newer wider", "low lower"],
        mojo_trainers.BpeTrainer(**trainer_args),
    )
    assert repeated.get_vocab() == expanded.get_vocab()
    assert repeated.model.merges == expanded.model.merges


def test_simd_tail_paths_match_upstream(bpe_pair, unigram_pair):
    for ours, theirs in (bpe_pair, unigram_pair):
        assert_encoding_equal(ours.encode("ababaab"), theirs.encode("ababaab"))


def test_large_count_table_parallel_threshold():
    token_counts, pair_counts = mojo_trainers._counts(
        [[index & 1 for index in range(20_003)]], [3], 2
    )
    assert token_counts == [30_006, 30_003]
    assert pair_counts == {(0, 1): 30_003, (1, 0): 30_003}


def test_small_count_table_serial_threshold():
    token_counts, pair_counts = mojo_trainers._counts(
        [[0, 1, 0], [1, 1]], [2, 3], 2
    )
    assert token_counts == [4, 8]
    assert pair_counts == {(0, 1): 2, (1, 0): 2, (1, 1): 3}


def test_ffi_helpers_reject_narrowing_and_invalid_buffers():
    with pytest.raises(OverflowError):
        mojo_lib.i64([1 << 63])
    with pytest.raises(TypeError):
        mojo_lib.i64([1.5])
    with pytest.raises(TypeError):
        mojo_lib.addr(np.zeros(1, dtype=np.int32), np.int64)
    with pytest.raises(ValueError):
        mojo_lib.addr(np.zeros((2, 2), dtype=np.int64), np.int64)
    with pytest.raises(RuntimeError):
        mojo_lib.checked_count(3, 2, "test kernel")


def test_u32_buffer_is_zero_copy_and_ffi_compatible():
    array = mojo_lib.u32("token")
    assert not array.flags.owndata
    assert mojo_lib.addr(array, np.uint32) == array.ctypes.data


def test_count_boundary_validation():
    with pytest.raises(ValueError):
        mojo_trainers._counts([[0]], [], 1)
    with pytest.raises(ValueError):
        mojo_trainers._counts([[1]], [1], 1)
    with pytest.raises(OverflowError):
        mojo_trainers._counts([[0, 0]], [1 << 62], 1)


CORPUS = ["low lower lowest", "newer wider", "low low"]


def test_bpe_training_exact_upstream_parity():
    ours, theirs = paired(
        mojo_models.BPE(unk_token="<unk>"),
        upstream_models.BPE(unk_token="<unk>"),
        pre="whitespace",
    )
    ours.train_from_iterator(
        CORPUS,
        mojo_trainers.BpeTrainer(
            vocab_size=20, special_tokens=["<unk>"], show_progress=False
        ),
    )
    theirs.train_from_iterator(
        CORPUS,
        upstream_trainers.BpeTrainer(
            vocab_size=20, special_tokens=["<unk>"], show_progress=False
        ),
    )
    assert ours.get_vocab() == theirs.get_vocab()
    for text in ("low lower newest", "wider", "xyz"):
        assert_encoding_equal(ours.encode(text), theirs.encode(text))


@pytest.mark.parametrize("vocab_size", [20, 25, 35])
def test_wordpiece_training_behavior(vocab_size):
    ours, theirs = paired(
        mojo_models.WordPiece(unk_token="[UNK]"),
        upstream_models.WordPiece(unk_token="[UNK]"),
        pre="whitespace",
    )
    ours.train_from_iterator(
        CORPUS,
        mojo_trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            show_progress=False,
        ),
    )
    theirs.train_from_iterator(
        CORPUS,
        upstream_trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            show_progress=False,
        ),
    )
    ours.decoder = mojo_decoders.WordPiece()
    theirs.decoder = upstream_decoders.WordPiece()
    for text in ("low lower newest", "wider", "xyz"):
        ours_encoding = ours.encode(text)
        theirs_encoding = theirs.encode(text)
        assert ours.decode(ours_encoding.ids) == theirs.decode(theirs_encoding.ids)
        assert all(token_id in ours.model._id_to_token for token_id in ours_encoding.ids)
        assert all(0 <= start <= end <= len(text) for start, end in ours_encoding.offsets)


def test_unigram_training_produces_complete_working_model():
    tokenizer = mt.Tokenizer(mojo_models.Unigram())
    tokenizer.pre_tokenizer = mojo_pre.Whitespace()
    tokenizer.train_from_iterator(
        CORPUS,
        mojo_trainers.UnigramTrainer(
            vocab_size=20,
            special_tokens=["<unk>"],
            unk_token="<unk>",
            show_progress=False,
        ),
    )
    assert tokenizer.get_vocab_size() == 20
    assert tokenizer.token_to_id("<unk>") == 0
    encoding = tokenizer.encode("low wider xyz")
    assert "".join(encoding.tokens[:2]).startswith("low")
    assert encoding.ids[-1] == 0
    assert encoding.offsets[-1] == (10, 13)


def test_lowercase_and_whitespace_pipeline_parity():
    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "!": 3}
    ours, theirs = paired(
        mojo_models.WordPiece(vocab),
        upstream_models.WordPiece(vocab),
        pre="whitespace",
        normalizer="lower",
    )
    assert_encoding_equal(ours.encode("Hello WORLD!"), theirs.encode("Hello WORLD!"))


@pytest.mark.parametrize(
    ("normalizer", "text"),
    [
        (mojo_normalizers.NFC(), "\u0315\u0300"),
        (mojo_normalizers.NFD(), "\u0315\u0300"),
        (mojo_normalizers.NFKC(), "Ａ"),
        (mojo_normalizers.NFKD(), "Ａ"),
    ],
)
def test_unicode_normalizers_match_python_and_preserve_length(normalizer, text):
    import unicodedata

    form = type(normalizer).__name__
    expected = unicodedata.normalize(form, text)
    assert len(expected) == len(text)
    assert normalizer.normalize_str(text) == expected


def test_character_count_changing_normalization_is_rejected():
    tokenizer = mt.Tokenizer(mojo_models.WordPiece({"[UNK]": 0, "é": 1}))
    tokenizer.normalizer = mojo_normalizers.NFC()
    with pytest.raises(NotImplementedError, match="character count"):
        tokenizer.encode("e\u0301")


def test_encode_pair_and_batch_parity(wordpiece_pair):
    ours, theirs = wordpiece_pair
    assert_encoding_equal(
        ours.encode("hello", "world"), theirs.encode("hello", "world")
    )
    ours_batch = ours.encode_batch(["hello", "wider", "unknown"])
    theirs_batch = theirs.encode_batch(["hello", "wider", "unknown"])
    for left, right in zip(ours_batch, theirs_batch):
        assert_encoding_equal(left, right)


def test_pretokenized_input_parity(wordpiece_pair):
    ours, theirs = wordpiece_pair
    assert_encoding_equal(
        ours.encode(["hello", "wider"], is_pretokenized=True),
        theirs.encode(["hello", "wider"], is_pretokenized=True),
    )


def test_wordpiece_decoder_parity():
    tokens = ["hello", "##s", "world", "!"]
    ours = mojo_decoders.WordPiece()
    theirs = upstream_decoders.WordPiece()
    assert ours.decode(tokens) == theirs.decode(tokens)


def test_fuse_decoder():
    assert mojo_decoders.Fuse().decode(["low", "er"]) == "lower"


def test_encoding_alignment_helpers(wordpiece_pair):
    ours, _ = wordpiece_pair
    encoding = ours.encode("hello wider")
    assert encoding.char_to_token(1) == 0
    assert encoding.token_to_chars(1) == (6, 7)
    assert encoding.word_to_tokens(1) == (1, 4)
    assert encoding.word_to_chars(1) == (6, 11)


def test_tokenizer_json_roundtrip(tmp_path):
    tokenizer = mt.Tokenizer(
        mojo_models.BPE(
            {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
            [("a", "b")],
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = mojo_pre.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    restored = mt.Tokenizer.from_file(str(path))
    assert restored.encode("ab x").ids == tokenizer.encode("ab x").ids
    assert json.loads(restored.to_str())["model"]["type"] == "BPE"


def test_bpe_model_file_roundtrip(tmp_path):
    model = mojo_models.BPE(
        {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
        [("a", "b")],
        unk_token="<unk>",
    )
    vocab_path, merges_path = model.save(str(tmp_path), "tiny")
    restored = mojo_models.BPE.from_file(
        vocab_path, merges_path, unk_token="<unk>"
    )
    assert restored.tokenize("ab") == model.tokenize("ab")


def test_wordpiece_model_file_roundtrip(tmp_path):
    model = mojo_models.WordPiece({"[UNK]": 0, "a": 1, "##b": 2})
    (path,) = model.save(str(tmp_path), "tiny")
    restored = mojo_models.WordPiece.from_file(path)
    assert restored.tokenize("ab") == model.tokenize("ab")


def test_model_tokenize_matches_upstream():
    vocab = {"[UNK]": 0, "a": 1, "##b": 2}
    ours = mojo_models.WordPiece(vocab).tokenize("ab")
    theirs = upstream_models.WordPiece(vocab).tokenize("ab")
    assert [(t.id, t.value, t.offsets) for t in ours] == [
        (t.id, t.value, t.offsets) for t in theirs
    ]
