"""Sparse column supports of the implicit LFSR generator G (shape K by N)."""

from functools import lru_cache

import numpy as np

from .lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms


@lru_cache(maxsize=16)
def _column_supports(n, terms):
    k = terms[-1]
    # Python integer masks work beyond 64 bits. Only K recent masks are retained.
    register = [0] * k
    offsets, indices = [0], []
    for column in range(n):
        if column < k:
            mask = 1 << column
        else:
            mask = 0
            for tap in terms[:-1]:
                mask ^= register[(column - k + tap) % k]
        register[column % k] = mask
        while mask:
            low = mask & -mask
            indices.append(low.bit_length() - 1)
            mask ^= low
        offsets.append(len(indices))
    offsets = np.asarray(offsets, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    offsets.setflags(write=False)
    indices.setflags(write=False)
    return offsets, indices


def column_supports(n, terms=DEFAULT_TERMS):
    """Return read-only CSC-style (offsets, x_indices), without dense G or A.

    Storage is O(K + N + nnz(G)); G may become dense for long observations.
    This graph represents encoding equations, not shifted recurrence checks.
    """
    return _column_supports(integer_parameter(n, "N", 1), polynomial_terms(terms))
