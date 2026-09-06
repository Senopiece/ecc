"""Sparse binary LFSR recurrence; no matrices, period search, or word-size limit."""

from numbers import Integral

import numpy as np
from numba import njit

DEFAULT_TERMS = (0, 2, 5)


def polynomial_terms(terms=DEFAULT_TERMS):
    """Validate representation only. Primitivity is the caller's assumption."""
    terms = tuple(terms)
    if any(not isinstance(p, Integral) or isinstance(p, bool) for p in terms):
        raise ValueError("Polynomial exponents must be integers")
    if len(terms) < 2 or len(set(terms)) != len(terms):
        raise ValueError("Supply distinct nonzero terms, including 0 and K")
    terms = tuple(sorted(int(p) for p in terms))
    if terms[0] != 0 or terms[-1] < 1:
        raise ValueError("Polynomial must have a constant term and positive degree")
    return terms


def integer_parameter(value, name, minimum=0):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@njit(cache=True)
def _encode(bits, n, taps, k):
    y = np.empty(n, dtype=np.uint8)
    for i in range(min(k, n)):
        y[i] = bits[i]
    # The output prefix doubles as the shift-register history. No extra K-vector
    # is needed, and this works for K > 64 without packed-integer overflow.
    for i in range(k, n):
        bit = np.uint8(0)
        for tap in taps:
            bit ^= y[i - k + tap]
        y[i] = bit
    return y


def encode(bits, n=16, terms=DEFAULT_TERMS):
    """Encode in O(K + N*w) time and O(N) output space, w = number of taps.

    y[:K] = bits and y[j+K] = XOR(y[j+p] for p in terms[:-1]).
    This equals x[e_0, A e_0, ...] for the polynomial-basis companion convention.
    """
    terms = polynomial_terms(terms)
    n = integer_parameter(n, "N", 1)
    bits = np.asarray(bits)
    if bits.ndim != 1 or len(bits) != terms[-1] or not np.all((bits == 0) | (bits == 1)):
        raise ValueError("x must contain exactly K binary values")
    return _encode(
        np.ascontiguousarray(bits, dtype=np.uint8),
        n,
        np.asarray(terms[:-1], dtype=np.int64),
        terms[-1],
    )


def max_availability(n, terms=DEFAULT_TERMS):
    """Maximum overlap of check-membership intervals; O(w log w), no N-array."""
    terms = polynomial_terms(terms)
    count = max(0, n - terms[-1])
    events = {}
    for p in terms:
        events[p] = events.get(p, 0) + 1
        events[p + count] = events.get(p + count, 0) - 1
    active = maximum = 0
    for p in sorted(events):
        active += events[p]
        maximum = max(maximum, active)
    return maximum
