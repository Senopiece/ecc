"""avail-softmajvote: synchronous addition of check-to-bit soft opinions."""

import numpy as np
from numba import njit

from ..lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms
from ._adt import NormalizedMinSum, SoftXor, SqrtSign, Tanh


def _softxor_options(softxor: SoftXor):
    """Lower the configuration to scalar arguments for the compiled kernel."""
    match softxor:
        case Tanh():
            return 0, 1.0
        case NormalizedMinSum(coefficient=alpha):
            return 1, alpha
        case SqrtSign():
            return 2, 1.0
        case _:
            raise TypeError("softxor must be Tanh(), NormalizedMinSum(...), or SqrtSign()")


@njit(cache=True, inline="always")
def _box_plus(a, b):
    # +infinity is the identity for the empty prefix/suffix of an XOR.
    if a == np.inf:
        return b
    if b == np.inf:
        return a
    if a == 0.0 or b == 0.0:
        return 0.0
    magnitude = min(abs(a), abs(b))
    sign = 1.0 if (a > 0) == (b > 0) else -1.0
    return sign * magnitude + np.log1p(np.exp(-abs(a + b))) - np.log1p(np.exp(-abs(a - b)))


@njit(cache=True)
def _iterate(initial, terms, iterations, mode, alpha):
    n = len(initial)
    width = len(terms)
    count = max(0, n - terms[-1])
    history = np.empty((iterations + 1, n), dtype=np.float64)
    history[0] = initial
    prefix = np.empty(width, dtype=np.float64)
    for t in range(iterations):
        previous = history[t]
        updated = history[t + 1]
        updated[:] = previous
        for start in range(count):
            if mode == 1:
                # Two minima and total sign give every excluded-target opinion
                # in O(w). Apply normalization once, not at each binary fold.
                minimum = second = np.inf
                negatives = zeros = 0
                for p in terms:
                    value = previous[start + p]
                    magnitude = abs(value)
                    negatives += value < 0
                    zeros += value == 0
                    if magnitude < minimum:
                        second = minimum
                        minimum = magnitude
                    elif magnitude < second:
                        second = magnitude
                for p in terms:
                    bit = start + p
                    value = previous[bit]
                    if zeros - (value == 0) > 0:
                        continue
                    magnitude = second if abs(value) == minimum else minimum
                    sign = -1.0 if (negatives - (value < 0)) % 2 else 1.0
                    updated[bit] += alpha * sign * magnitude
                continue
            if mode == 2:
                # Symmetric m-th root, m = width - 1. Log magnitudes avoid
                # overflow/underflow of the product. Reuse scratch storage.
                log_sum = 0.0
                negatives = zeros = 0
                for j in range(width):
                    value = previous[start + terms[j]]
                    negatives += value < 0
                    zeros += value == 0
                    prefix[j] = np.log(abs(value)) if value != 0 else 0.0
                    log_sum += prefix[j]
                for target in range(width):
                    bit = start + terms[target]
                    value = previous[bit]
                    if zeros - (value == 0) > 0:
                        continue
                    sign = -1.0 if (negatives - (value < 0)) % 2 else 1.0
                    log_mean = (log_sum - prefix[target]) / (width - 1)
                    updated[bit] += sign * np.exp(log_mean)
                continue
            prefix[0] = np.inf
            # The full-check XOR is never used by any leave-one-out message.
            for j in range(width - 1):
                prefix[j + 1] = _box_plus(prefix[j], previous[start + terms[j]])
            suffix = np.inf
            for j in range(width - 1, -1, -1):
                bit = start + terms[j]
                # Exclude the target using prefix/suffix folds: O(w), not O(w^2).
                updated[bit] += _box_plus(prefix[j], suffix)
                if j > 0:
                    suffix = _box_plus(previous[bit], suffix)
        for bit in range(n):
            if not np.isfinite(updated[bit]):
                raise FloatingPointError("LLR overflow: reduce T or input magnitudes")
    return history


def decode(
    initial,
    terms=DEFAULT_TERMS,
    iterations=20,
    *,
    softxor: SoftXor = Tanh(),
) -> np.ndarray:
    """Return all T+1 states; checks are implicit shifts of sparse exponents.

    All modes take O(T*(N + max(0,N-K)*w)) time. Storage O((T+1)*N + w).
    No truth, channel reinjection, clipping, damping, or early stopping.
    """
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    mode, alpha = _softxor_options(softxor)
    initial = np.asarray(initial, dtype=np.float64)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("Initial LLRs must be a nonempty finite vector")
    return _iterate(
        np.ascontiguousarray(initial),
        np.asarray(terms, dtype=np.int64),
        iterations,
        mode,
        alpha,
    )
