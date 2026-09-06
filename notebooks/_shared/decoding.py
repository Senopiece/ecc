"""Explicit, snapshot-based synchronous additive soft-XOR decoding."""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms

SOFTXOR_MODES = ("tanh", "normalized-min-sum", "sqrt-sign")


def _softxor_options(softxor, coefficient):
    if softxor not in SOFTXOR_MODES:
        raise ValueError(f"softxor must be one of {SOFTXOR_MODES}")
    mode = SOFTXOR_MODES.index(softxor)
    alpha = float(coefficient) if mode == 1 else 1.0
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Normalized min-sum coefficient must be between 0 and 1")
    return mode, alpha


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
                    updated[bit] += sign * np.exp(log_mean - np.log(2.0))
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


def iterate(initial, terms=DEFAULT_TERMS, iterations=20, *, softxor="tanh", coefficient=0.8):
    """Return all T+1 states; checks are implicit shifts of sparse exponents.

    All modes take O(T*(N + max(0,N-K)*w)) time. Storage O((T+1)*N + w).
    No truth, channel reinjection, clipping, damping, or early stopping.
    """
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    mode, alpha = _softxor_options(softxor, coefficient)
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


@njit(cache=True)
def _metrics(history, truth, terms):
    steps, n = history.shape
    count = max(0, n - terms[-1])
    scores = np.empty((5, steps), dtype=np.float64)
    for t in range(steps):
        errors = erasures = loss = signed_sum = 0.0
        for i in range(n):
            margin = history[t, i] if truth[i] == 0 else -history[t, i]
            if margin < 0:
                errors += 1
            elif margin == 0:
                errors += 0.5
                erasures += 1
            loss += (max(0.0, -margin) + np.log1p(np.exp(-abs(margin)))) / n
            signed_sum += margin / n
        violated = 0
        for start in range(count):
            parity = False
            for p in terms:
                parity ^= history[t, start + p] < 0
            violated += parity
        scores[0, t] = errors / n
        scores[1, t] = erasures / n
        scores[2, t] = violated / count if count else np.nan
        scores[3, t] = loss
        scores[4, t] = signed_sum
    return scores


@dataclass(frozen=True)
class DecodeResult:
    """One completed run, detached from any subsequent edits in the widget."""

    history: np.ndarray
    truth: np.ndarray
    terms: tuple[int, ...]
    scores: dict[str, np.ndarray]
    softxor: str
    coefficient: float | None


def decode(
    initial,
    truth,
    terms=DEFAULT_TERMS,
    iterations=20,
    *,
    softxor="tanh",
    coefficient=0.8,
):
    """Explicit entry point: decode a snapshot and compute diagnostics once."""
    terms = polynomial_terms(terms)
    initial = np.array(initial, dtype=np.float64, copy=True)
    truth = np.asarray(truth)
    if truth.shape != initial.shape or not np.all((truth == 0) | (truth == 1)):
        raise ValueError("Truth must be a binary vector matching the observation")
    truth = np.array(truth, dtype=np.uint8, copy=True)
    mode, alpha = _softxor_options(softxor, coefficient)
    history = iterate(initial, terms, iterations, softxor=softxor, coefficient=alpha)
    values = _metrics(history, truth, np.asarray(terms, dtype=np.int64))
    names = (
        "Sign error rate (ties = 1/2)",
        "Erasure fraction",
        "Unsatisfied checks",
        "Mean logistic loss",
        "Mean signed margin",
    )
    history.flags.writeable = False
    truth.flags.writeable = False
    values.flags.writeable = False
    return DecodeResult(
        history,
        truth,
        terms,
        dict(zip(names, values, strict=True)),
        softxor,
        alpha if mode == 1 else None,
    )
