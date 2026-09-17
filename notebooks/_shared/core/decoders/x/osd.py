"""Packed GF(2) most-reliable-basis OSD for arbitrary binary generators."""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from numba import njit

from ...lfsr import integer_parameter


@dataclass(frozen=True)
class OSDConfig:
    order: int = 1
    search_bits: int = 0  # 0 = full MRB; positive = least reliable MRB subset.
    max_candidates: int = 4096

    def __post_init__(self):
        integer_parameter(self.order, "OSD order")
        if self.order > 2:
            raise ValueError("This low-order OSD implementation supports orders 0, 1, 2")
        integer_parameter(self.search_bits, "search_bits")
        integer_parameter(self.max_candidates, "max_candidates", 1)


@lru_cache(maxsize=64)
def patterns(dimension, config):
    width = min(dimension, config.search_bits or dimension)
    result = [(-1, -1)]
    for order in range(1, config.order + 1):
        for combo in combinations(range(width), order):
            if len(result) >= config.max_candidates:
                break
            result.append((combo[0], combo[1] if order == 2 else -1))
    result = np.asarray(result, dtype=np.int64)
    result.setflags(write=False)
    return result


@njit(cache=True, inline="always")
def _bit(row, column):
    return (row[column >> 6] >> np.uint64(column & 63)) & np.uint64(1)


@njit(cache=True, nogil=True)
def decode_packed(llr, generator_rows, error_patterns):
    """Return (packed full hard message, rank_ok). No phase or payload knowledge.

    Rows are uint64-packed across channel coordinates. Row operations also act
    on a packed identity transform, preserving the original message basis.
    Candidate score is weighted Hamming distance to hard decisions.
    """
    dimension, length = len(generator_rows), len(llr)
    words, message_words = (length + 63) // 64, (dimension + 63) // 64
    result = np.zeros(message_words, dtype=np.uint64)
    if length < dimension:
        return result, False
    matrix = np.zeros((dimension, words + message_words), dtype=np.uint64)
    matrix[:, :words] = generator_rows[:, :words]
    for row in range(dimension):
        matrix[row, words + (row >> 6)] = np.uint64(1) << np.uint64(row & 63)
    reliability = np.abs(llr)
    order = np.argsort(-reliability)
    pivots = np.empty(dimension, dtype=np.int64)
    rank = 0
    for column in order:
        pivot = rank
        while pivot < dimension and _bit(matrix[pivot], column) == 0:
            pivot += 1
        if pivot == dimension:
            continue
        for word in range(words + message_words):
            matrix[rank, word], matrix[pivot, word] = matrix[pivot, word], matrix[rank, word]
        for row in range(dimension):
            if row != rank and _bit(matrix[row], column):
                for word in range(words + message_words):
                    matrix[row, word] ^= matrix[rank, word]
        pivots[rank] = column
        rank += 1
        if rank == dimension:
            break
    if rank != dimension:
        return result, False
    base = np.zeros(words + message_words, dtype=np.uint64)
    for row in range(dimension):
        if llr[pivots[row]] < 0:
            base ^= matrix[row]
    # Eight-bit weighted-distance lookup: packed candidates need only N/8 loads.
    byte_count = (length + 7) // 8
    weights = np.zeros((byte_count, 256), dtype=np.float64)
    hard = np.zeros(byte_count, dtype=np.uint64)
    for byte in range(byte_count):
        for bit in range(8):
            column = 8 * byte + bit
            if column < length and llr[column] < 0:
                hard[byte] |= np.uint64(1) << np.uint64(bit)
        for value in range(1, 256):
            low = value & -value
            bit = 0
            while (1 << bit) != low:
                bit += 1
            column = 8 * byte + bit
            weights[byte, value] = weights[byte, value ^ low]
            if column < length:
                weights[byte, value] += reliability[column]
    best = np.inf
    best_a = best_b = -1
    for pattern in error_patterns:
        a = dimension - 1 - pattern[0] if pattern[0] >= 0 else -1
        b = dimension - 1 - pattern[1] if pattern[1] >= 0 else -1
        score = 0.0
        for byte in range(byte_count):
            word = byte >> 3
            packed = base[word]
            if a >= 0:
                packed ^= matrix[a, word]
            if b >= 0:
                packed ^= matrix[b, word]
            error = ((packed >> np.uint64((byte & 7) * 8)) & np.uint64(255)) ^ hard[byte]
            score += weights[byte, error]
            if score >= best:
                break
        if score < best:
            best, best_a, best_b = score, a, b
            if best == 0:
                break
    for word in range(message_words):
        result[word] = base[words + word]
        if best_a >= 0:
            result[word] ^= matrix[best_a, words + word]
        if best_b >= 0:
            result[word] ^= matrix[best_b, words + word]
    return result, True
