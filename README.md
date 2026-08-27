# mojo-tokenizers

BPE, WordPiece, and Unigram tokenization with the compute-heavy loops written in
[Mojo](https://www.modular.com/mojo) and a Python API shaped like Hugging Face
[`tokenizers`](https://github.com/huggingface/tokenizers).

This is a standalone implementation: it does not call the Rust `tokenizers` package at
runtime. That package is present in the Pixi development environment only for parity
tests and benchmarks.

```python
from mojo_tokenizers import Tokenizer
from mojo_tokenizers.models import BPE
from mojo_tokenizers.pre_tokenizers import Whitespace
from mojo_tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train_from_iterator(
    ["low lower lowest", "newer wider", "low low"],
    BpeTrainer(vocab_size=20, special_tokens=["<unk>"], show_progress=False),
)

encoding = tokenizer.encode("low newest")
print(encoding.tokens)   # ['low', 'new', 'es', 't']
print(encoding.offsets)  # [(0, 3), (4, 7), (7, 9), (9, 10)]
```

## Coverage

| area | implemented |
| --- | --- |
| Models | `models.BPE`, `models.WordPiece`, `models.Unigram` |
| Trainers | `BpeTrainer`, `WordPieceTrainer`, SentencePiece-style EM `UnigramTrainer` |
| Pipeline | `Tokenizer.encode`, `encode_batch`, paired and pretokenized input, `decode`, training from iterators |
| Results | `Encoding` IDs, tokens, offsets, word/type/sequence IDs, masks, alignment helpers |
| Pre-tokenizers | `Whitespace`, `WhitespaceSplit` |
| Normalizers | lowercase and Unicode normalization when character count is preserved |
| Decoders | `WordPiece`, `Fuse` |
| Persistence | tokenizer JSON round trips; BPE and WordPiece vocabulary files |

Fixed-vocabulary encoding is behaviorally parity-tested against `tokenizers 0.23.1`,
including Unicode offsets, unknown spans, BPE prefixes/suffixes and byte fallback,
WordPiece longest match, and Unigram Viterbi segmentation. BPE training produces the
same vocabulary, merge order, IDs, and encodings as upstream on the parity corpus.
WordPiece training is tested to produce valid encodings and the same decoded text as
upstream. Its vocabulary and segmentation can differ when equally scored merges are
available because upstream's hash iteration order is not stable.

The Unigram trainer is a real EM trainer: it seeds substring candidates, runs
forward/backward expected-count updates, preserves the character alphabet, and prunes
by expected use. Its trained scores and selected vocabulary are not byte-for-byte
identical to SentencePiece or Hugging Face's seed/pruning heuristics. Encoding an
existing scored Unigram vocabulary is parity-tested against upstream.

Not covered are added-token splitting rules, post-processors, truncation/padding,
ByteLevel pre-tokenization, BPE dropout, Unigram sampling and Unigram byte fallback.
Normalizers that change the number of characters are rejected because reproducing
upstream's original-text offset alignment requires a normalization alignment map.

## Install

Clone the repository, then install the pinned Mojo nightly and development dependencies
with Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-tokenizers.so`. The Python bridge also rebuilds
the library when a Mojo source is newer than the shared object.

The package is currently intended to run from this source checkout through Pixi; a
prebuilt wheel is not published.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic. Times are the best of five runs for encoding and three for training.
The ratio is upstream time divided by mojo-tokenizers time, so values below 1 mean this
port is slower.

| case | mojo-tokenizers | tokenizers 0.23.1 | ratio |
| --- | ---: | ---: | ---: |
| BPE encode (0.32M chars) | 66.63 ms | 70.03 ms | 1.05x |
| WordPiece encode (0.32M chars) | 67.71 ms | 77.30 ms | 1.14x |
| Unigram encode (0.32M chars) | 148.93 ms | 167.57 ms | 1.13x |
| BPE train (4k lines, vocab 160) | 11.84 ms | 13.01 ms | 1.10x |
| WordPiece train (4k lines, vocab 160) | 9.92 ms | 12.26 ms | 1.24x |
| Unigram train (4k lines, vocab 160) | 10.50 ms | 14.40 ms | 1.37x |

BPE, WordPiece, and Unigram cache bounded per-segment encodings, corpus preparation
aggregates repeated input lines before pre-tokenization, and small trainer states stay
serial to avoid FFI and thread-launch overhead. Large pair-table clears use a parallel
threshold; BPE scratch copies, pair-table initialization, and Unigram dynamic-programming
initialization use SIMD with scalar tails. UTF-32 input views cross the FFI boundary
without an additional NumPy copy.

There is no GPU path. These kernels are dominated by branch-heavy hash probes, short
string comparisons, and low-intensity buffer initialization, all below the arithmetic
intensity where transfer and launch costs are justified.

## How it works

Python owns every allocation. Text and vocabulary pieces are encoded as contiguous
UTF-32 NumPy buffers; vocabulary strings are stored in one flat buffer with offset and
length arrays. Mojo uses open-addressed hash tables for token and merge lookup.

The ctypes boundary passes buffers as 64-bit integer addresses. Each exported
`@export("...")` function reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside an `abi("C")` wrapper. One shared
library contains four main operations:

- ranked, non-overlapping BPE merge application;
- greedy longest-match WordPiece search;
- scored Unigram lattice Viterbi decoding and backtracking;
- weighted token and adjacent-pair counting for BPE/WordPiece trainers.

IDs, character offsets, scores, dynamic-programming state, and scratch memory are all
caller-owned contiguous arrays. Mojo performs no cross-FFI allocation, so there is no
foreign lifetime or deallocator to manage.

## License

MIT
