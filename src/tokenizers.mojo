"""Compute kernels for subword tokenization and training."""

from max.algorithm import parallelize
from std.sys.info import simd_width_of

comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def copy_i64(
    destination: IPtr,
    destination_start: Int,
    source: IPtr,
    source_start: Int,
    count: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= count:
        destination.store(destination_start + i, source.load[width=W](source_start + i))
        i += W
    while i < count:
        destination[destination_start + i] = source[source_start + i]
        i += 1


def fill_f64(pointer: FPtr, start: Int, count: Int, value: Float64):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vector = SIMD[DType.float64, W](value)
    while i + W <= count:
        pointer.store(start + i, vector)
        i += W
    while i < count:
        pointer[start + i] = value
        i += 1


def fill_i64(pointer: IPtr, start: Int, count: Int, value: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var vector = SIMD[DType.int, W](value)
    while i + W <= count:
        pointer.store(start + i, vector)
        i += W
    while i < count:
        pointer[start + i] = value
        i += 1


def pair_slot(
    left: Int, right: Int, keys_left: IPtr, keys_right: IPtr, cap: Int
) -> Int:
    var h = UInt64(left) * 11400714819323198485
    h = h ^ (UInt64(right) * 14029467366897019727)
    var slot = Int(h & UInt64(cap - 1))
    while keys_left[slot] != -1:
        if keys_left[slot] == left and keys_right[slot] == right:
            return slot
        slot = (slot + 1) & (cap - 1)
    return slot


@export("mt_count_tokens_pairs")
def mt_count_tokens_pairs(
    ids_addr: Int,
    offsets_addr: Int,
    freqs_addr: Int,
    n_words: Int,
    token_counts_addr: Int,
    n_tokens: Int,
    pair_left_addr: Int,
    pair_right_addr: Int,
    pair_counts_addr: Int,
    cap: Int,
) abi("C") -> Int:
    var ids = IPtr(unsafe_from_address=ids_addr)
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var freqs = IPtr(unsafe_from_address=freqs_addr)
    var token_counts = IPtr(unsafe_from_address=token_counts_addr)
    var pair_left = IPtr(unsafe_from_address=pair_left_addr)
    var pair_right = IPtr(unsafe_from_address=pair_right_addr)
    var pair_counts = IPtr(unsafe_from_address=pair_counts_addr)
    fill_i64(token_counts, 0, n_tokens, 0)
    comptime CLEAR_CHUNK = 16_384

    @__copy_capture(pair_left, pair_right, pair_counts, cap)
    @__parameter
    def clear_chunk(chunk: Int):
        var start = chunk * CLEAR_CHUNK
        var end = min(cap, start + CLEAR_CHUNK)
        fill_i64(pair_left, start, end - start, -1)
        fill_i64(pair_right, start, end - start, -1)
        fill_i64(pair_counts, start, end - start, 0)

    var clear_chunks = (cap + CLEAR_CHUNK - 1) // CLEAR_CHUNK
    if cap >= 65_536:
        parallelize[clear_chunk](clear_chunks, min(clear_chunks, 8))
    else:
        for chunk in range(clear_chunks):
            clear_chunk(chunk)
    var unique = 0
    for w in range(n_words):
        var start = Int(offsets[w])
        var end = Int(offsets[w + 1])
        var freq = freqs[w]
        for i in range(start, end):
            token_counts[Int(ids[i])] += freq
        for i in range(start, end - 1):
            var left = Int(ids[i])
            var right = Int(ids[i + 1])
            var slot = pair_slot(left, right, pair_left, pair_right, cap)
            if pair_left[slot] == -1:
                pair_left[slot] = left
                pair_right[slot] = right
                unique += 1
            pair_counts[slot] += freq
    return unique


def merge_rank(
    left: Int,
    right: Int,
    table_left: IPtr,
    table_right: IPtr,
    table_rank: IPtr,
    cap: Int,
) -> Int:
    var slot = pair_slot(left, right, table_left, table_right, cap)
    if table_left[slot] == -1:
        return -1
    return Int(table_rank[slot])


def merge_result(
    left: Int,
    right: Int,
    table_left: IPtr,
    table_right: IPtr,
    table_result: IPtr,
    cap: Int,
) -> Int:
    var slot = pair_slot(left, right, table_left, table_right, cap)
    if table_left[slot] == -1:
        return -1
    return Int(table_result[slot])


@export("mt_bpe_encode")
def mt_bpe_encode(
    ids_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
    segment_offsets_addr: Int,
    n_segments: Int,
    table_left_addr: Int,
    table_right_addr: Int,
    table_rank_addr: Int,
    table_result_addr: Int,
    table_cap: Int,
    unk_id: Int,
    fuse_unk: Int,
    result_ids_addr: Int,
    result_starts_addr: Int,
    result_ends_addr: Int,
    work_ids_addr: Int,
    work_starts_addr: Int,
    work_ends_addr: Int,
) abi("C") -> Int:
    var ids = IPtr(unsafe_from_address=ids_addr)
    var starts = IPtr(unsafe_from_address=starts_addr)
    var ends = IPtr(unsafe_from_address=ends_addr)
    var segment_offsets = IPtr(unsafe_from_address=segment_offsets_addr)
    var table_left = IPtr(unsafe_from_address=table_left_addr)
    var table_right = IPtr(unsafe_from_address=table_right_addr)
    var table_rank = IPtr(unsafe_from_address=table_rank_addr)
    var table_result = IPtr(unsafe_from_address=table_result_addr)
    var result_ids = IPtr(unsafe_from_address=result_ids_addr)
    var result_starts = IPtr(unsafe_from_address=result_starts_addr)
    var result_ends = IPtr(unsafe_from_address=result_ends_addr)
    var work_ids = IPtr(unsafe_from_address=work_ids_addr)
    var work_starts = IPtr(unsafe_from_address=work_starts_addr)
    var work_ends = IPtr(unsafe_from_address=work_ends_addr)
    var result_count = 0
    for segment in range(n_segments):
        var source_start = Int(segment_offsets[segment])
        var length = Int(segment_offsets[segment + 1]) - source_start
        copy_i64(work_ids, 0, ids, source_start, length)
        copy_i64(work_starts, 0, starts, source_start, length)
        copy_i64(work_ends, 0, ends, source_start, length)
        while length > 1:
            var best_rank = -1
            for i in range(length - 1):
                var rank = merge_rank(
                    Int(work_ids[i]),
                    Int(work_ids[i + 1]),
                    table_left,
                    table_right,
                    table_rank,
                    table_cap,
                )
                if rank >= 0 and (best_rank < 0 or rank < best_rank):
                    best_rank = rank
            if best_rank < 0:
                break
            var read = 0
            var write = 0
            while read < length:
                if read + 1 < length:
                    var rank = merge_rank(
                        Int(work_ids[read]),
                        Int(work_ids[read + 1]),
                        table_left,
                        table_right,
                        table_rank,
                        table_cap,
                    )
                    if rank == best_rank:
                        work_ids[write] = merge_result(
                            Int(work_ids[read]),
                            Int(work_ids[read + 1]),
                            table_left,
                            table_right,
                            table_result,
                            table_cap,
                        )
                        work_starts[write] = work_starts[read]
                        work_ends[write] = work_ends[read + 1]
                        read += 2
                        write += 1
                        continue
                work_ids[write] = work_ids[read]
                work_starts[write] = work_starts[read]
                work_ends[write] = work_ends[read]
                read += 1
                write += 1
            length = write
        for i in range(length):
            var token_id = Int(work_ids[i])
            if (
                fuse_unk != 0
                and token_id == unk_id
                and result_count > 0
                and Int(result_ids[result_count - 1]) == unk_id
                and result_ends[result_count - 1] == work_starts[i]
            ):
                result_ends[result_count - 1] = work_ends[i]
            else:
                result_ids[result_count] = token_id
                result_starts[result_count] = work_starts[i]
                result_ends[result_count] = work_ends[i]
                result_count += 1
    return result_count


def hash_chars(chars: U32Ptr, start: Int, end: Int) -> UInt64:
    var h = UInt64(1469598103934665603)
    for i in range(start, end):
        h = (h ^ UInt64(chars[i])) * 1099511628211
    return h


def hash_prefixed(
    prefix: U32Ptr, prefix_len: Int, chars: U32Ptr, start: Int, end: Int
) -> UInt64:
    var h = UInt64(1469598103934665603)
    for i in range(prefix_len):
        h = (h ^ UInt64(prefix[i])) * 1099511628211
    for i in range(start, end):
        h = (h ^ UInt64(chars[i])) * 1099511628211
    return h


def vocab_match(
    chars: U32Ptr,
    start: Int,
    end: Int,
    prefix: U32Ptr,
    prefix_len: Int,
    vocab_chars: U32Ptr,
    vocab_offsets: IPtr,
    vocab_lengths: IPtr,
    table: IPtr,
    table_cap: Int,
) -> Int:
    var h = hash_prefixed(prefix, prefix_len, chars, start, end)
    var slot = Int(h & UInt64(table_cap - 1))
    while table[slot] != -1:
        var token_id = Int(table[slot])
        var expected = prefix_len + end - start
        if Int(vocab_lengths[token_id]) == expected:
            var base = Int(vocab_offsets[token_id])
            var same = True
            for i in range(prefix_len):
                if vocab_chars[base + i] != prefix[i]:
                    same = False
                    break
            if same:
                for i in range(end - start):
                    if vocab_chars[base + prefix_len + i] != chars[start + i]:
                        same = False
                        break
            if same:
                return token_id
        slot = (slot + 1) & (table_cap - 1)
    return -1


@export("mt_wordpiece_encode")
def mt_wordpiece_encode(
    chars_addr: Int,
    segment_starts_addr: Int,
    segment_ends_addr: Int,
    n_segments: Int,
    prefix_addr: Int,
    prefix_len: Int,
    vocab_chars_addr: Int,
    vocab_offsets_addr: Int,
    vocab_lengths_addr: Int,
    table_addr: Int,
    table_cap: Int,
    unk_id: Int,
    max_chars: Int,
    result_ids_addr: Int,
    result_starts_addr: Int,
    result_ends_addr: Int,
) abi("C") -> Int:
    var chars = U32Ptr(unsafe_from_address=chars_addr)
    var segment_starts = IPtr(unsafe_from_address=segment_starts_addr)
    var segment_ends = IPtr(unsafe_from_address=segment_ends_addr)
    var prefix = U32Ptr(unsafe_from_address=prefix_addr)
    var vocab_chars = U32Ptr(unsafe_from_address=vocab_chars_addr)
    var vocab_offsets = IPtr(unsafe_from_address=vocab_offsets_addr)
    var vocab_lengths = IPtr(unsafe_from_address=vocab_lengths_addr)
    var table = IPtr(unsafe_from_address=table_addr)
    var result_ids = IPtr(unsafe_from_address=result_ids_addr)
    var result_starts = IPtr(unsafe_from_address=result_starts_addr)
    var result_ends = IPtr(unsafe_from_address=result_ends_addr)
    var result_count = 0
    for segment in range(n_segments):
        var seg_start = Int(segment_starts[segment])
        var seg_end = Int(segment_ends[segment])
        var before = result_count
        var failed = seg_end - seg_start > max_chars
        var pos = seg_start
        while pos < seg_end and not failed:
            var candidate_end = seg_end
            var found = -1
            var use_prefix = prefix_len if pos > seg_start else 0
            while candidate_end > pos:
                found = vocab_match(
                    chars,
                    pos,
                    candidate_end,
                    prefix,
                    use_prefix,
                    vocab_chars,
                    vocab_offsets,
                    vocab_lengths,
                    table,
                    table_cap,
                )
                if found >= 0:
                    break
                candidate_end -= 1
            if found < 0:
                failed = True
                break
            result_ids[result_count] = found
            result_starts[result_count] = pos
            result_ends[result_count] = candidate_end
            result_count += 1
            pos = candidate_end
        if failed:
            result_count = before
            result_ids[result_count] = unk_id
            result_starts[result_count] = seg_start
            result_ends[result_count] = seg_end
            result_count += 1
    return result_count


def log_add(a: Float64, b: Float64) -> Float64:
    from std.math import exp, log

    if a < -1.0e290:
        return b
    if b < -1.0e290:
        return a
    if a < b:
        return b + log(1.0 + exp(a - b))
    return a + log(1.0 + exp(b - a))


@export("mt_unigram_encode")
def mt_unigram_encode(
    chars_addr: Int,
    segment_starts_addr: Int,
    segment_ends_addr: Int,
    n_segments: Int,
    vocab_chars_addr: Int,
    vocab_offsets_addr: Int,
    vocab_lengths_addr: Int,
    scores_addr: Int,
    table_addr: Int,
    table_cap: Int,
    unk_id: Int,
    unk_score: Float64,
    max_piece_chars: Int,
    dp_addr: Int,
    previous_addr: Int,
    previous_id_addr: Int,
    temp_ids_addr: Int,
    temp_starts_addr: Int,
    temp_ends_addr: Int,
    result_ids_addr: Int,
    result_starts_addr: Int,
    result_ends_addr: Int,
) abi("C") -> Int:
    var chars = U32Ptr(unsafe_from_address=chars_addr)
    var segment_starts = IPtr(unsafe_from_address=segment_starts_addr)
    var segment_ends = IPtr(unsafe_from_address=segment_ends_addr)
    var vocab_chars = U32Ptr(unsafe_from_address=vocab_chars_addr)
    var vocab_offsets = IPtr(unsafe_from_address=vocab_offsets_addr)
    var vocab_lengths = IPtr(unsafe_from_address=vocab_lengths_addr)
    var scores = FPtr(unsafe_from_address=scores_addr)
    var table = IPtr(unsafe_from_address=table_addr)
    var dp = FPtr(unsafe_from_address=dp_addr)
    var previous = IPtr(unsafe_from_address=previous_addr)
    var previous_id = IPtr(unsafe_from_address=previous_id_addr)
    var temp_ids = IPtr(unsafe_from_address=temp_ids_addr)
    var temp_starts = IPtr(unsafe_from_address=temp_starts_addr)
    var temp_ends = IPtr(unsafe_from_address=temp_ends_addr)
    var result_ids = IPtr(unsafe_from_address=result_ids_addr)
    var result_starts = IPtr(unsafe_from_address=result_starts_addr)
    var result_ends = IPtr(unsafe_from_address=result_ends_addr)
    var empty_prefix = chars
    var result_count = 0
    for segment in range(n_segments):
        var seg_start = Int(segment_starts[segment])
        var seg_end = Int(segment_ends[segment])
        fill_f64(dp, seg_start, seg_end - seg_start + 1, -1.0e300)
        for i in range(seg_start, seg_end + 1):
            previous[i] = -1
            previous_id[i] = -1
        dp[seg_start] = 0.0
        for pos in range(seg_start, seg_end):
            if dp[pos] < -1.0e290:
                continue
            var limit = min(seg_end, pos + max_piece_chars)
            var found_known = False
            for candidate_end in range(pos + 1, limit + 1):
                var token_id = vocab_match(
                    chars,
                    pos,
                    candidate_end,
                    empty_prefix,
                    0,
                    vocab_chars,
                    vocab_offsets,
                    vocab_lengths,
                    table,
                    table_cap,
                )
                if token_id >= 0 and token_id != unk_id:
                    found_known = True
                    var candidate_score = dp[pos] + scores[token_id]
                    if candidate_score > dp[candidate_end]:
                        dp[candidate_end] = candidate_score
                        previous[candidate_end] = pos
                        previous_id[candidate_end] = token_id
            if not found_known:
                var candidate_score = dp[pos] + unk_score
                if candidate_score > dp[pos + 1]:
                    dp[pos + 1] = candidate_score
                    previous[pos + 1] = pos
                    previous_id[pos + 1] = unk_id
        var cursor = seg_end
        var temp_count = 0
        while cursor > seg_start:
            var prev = Int(previous[cursor])
            if prev < seg_start:
                prev = cursor - 1
            temp_ids[temp_count] = previous_id[cursor]
            if temp_ids[temp_count] < 0:
                temp_ids[temp_count] = unk_id
            temp_starts[temp_count] = prev
            temp_ends[temp_count] = cursor
            cursor = prev
            temp_count += 1
        for reverse_i in range(temp_count):
            var i = temp_count - reverse_i - 1
            var token_id = Int(temp_ids[i])
            if (
                token_id == unk_id
                and result_count > 0
                and Int(result_ids[result_count - 1]) == unk_id
                and result_ends[result_count - 1] == temp_starts[i]
            ):
                result_ends[result_count - 1] = temp_ends[i]
            else:
                result_ids[result_count] = token_id
                result_starts[result_count] = temp_starts[i]
                result_ends[result_count] = temp_ends[i]
                result_count += 1
    return result_count
