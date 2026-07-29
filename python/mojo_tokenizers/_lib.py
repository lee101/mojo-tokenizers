"""ctypes bridge to the Mojo tokenizer kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-tokenizers.so")
BUILD = os.path.join(ROOT, "build", "build.sh")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mt_count_tokens_pairs": ([I] * 10, I),
    "mt_bpe_encode": ([I] * 18, I),
    "mt_wordpiece_encode": ([I] * 16, I),
    "mt_unigram_encode": (
        [I] * 11 + [F] + [I] * 10,
        I,
    ),
}

_library: ctypes.CDLL | None = None
_I64 = np.iinfo(np.int64)


def build(force: bool = False) -> str:
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(os.path.join(ROOT, "src"))
        for name in names
        if name.endswith(".mojo")
    ]
    stale = (
        not os.path.exists(LIB)
        or any(os.path.getmtime(source) > os.path.getmtime(LIB) for source in sources)
    )
    if force or stale:
        proc = subprocess.run(
            ["bash", BUILD], cwd=ROOT, capture_output=True, text=True, timeout=1800
        )
        if proc.returncode or not os.path.exists(LIB):
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def i64(values=(), *, size: int | None = None, fill: int | None = None) -> np.ndarray:
    if size is not None:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("array size must be a non-negative integer")
        array = np.empty(max(1, size), dtype=np.int64)
        if fill is not None:
            _check_i64(fill)
            array.fill(fill)
        return array
    checked = [_check_i64(value) for value in values]
    array = np.ascontiguousarray(checked, dtype=np.int64)
    return array if array.size else np.zeros(1, dtype=np.int64)


def f64(values=(), *, size: int | None = None) -> np.ndarray:
    if size is not None:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("array size must be a non-negative integer")
        return np.empty(max(1, size), dtype=np.float64)
    array = np.ascontiguousarray(values, dtype=np.float64)
    return array if array.size else np.zeros(1, dtype=np.float64)


def u32(text: str) -> np.ndarray:
    array = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    if array.size:
        return np.array(array, dtype=np.uint32, order="C", copy=True)
    return np.zeros(1, dtype=np.uint32)


def addr(array: np.ndarray, dtype: np.dtype | type) -> int:
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(f"FFI buffer must have dtype {expected}, got {array.dtype}")
    if array.ndim != 1 or not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned, contiguous one-dimensional arrays")
    if array.size < 1 or not array.ctypes.data:
        raise ValueError("FFI buffers must have a non-null allocation")
    return int(array.ctypes.data)


def power_of_two(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("value must be a positive integer")
    return 1 << max(1, value - 1).bit_length()


def checked_count(count: int, capacity: int, operation: str) -> int:
    count = int(count)
    if count < 0 or count > capacity:
        raise RuntimeError(
            f"{operation} returned invalid result length {count} for capacity {capacity}"
        )
    return count


def _check_i64(value) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    value = int(value)
    if value < _I64.min or value > _I64.max:
        raise OverflowError(f"integer {value} does not fit in int64")
    return value
