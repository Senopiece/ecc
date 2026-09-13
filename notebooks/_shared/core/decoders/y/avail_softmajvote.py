"""avail-softmajvote: synchronous addition of check-to-bit soft opinions."""

import numpy as np
from numba import njit

from ...lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms
from .._adt import SoftXor, Tanh
from .._softxor import _box_plus, _softxor_options


@njit(cache=True)
def _iterate(initial, terms, iterations, mode, alpha, return_history):
    n = len(initial)
    width = len(terms)
    count = max(0, n - terms[-1])
    history = np.empty((iterations + 1 if return_history else 2, n), dtype=np.float64)
    history[0] = initial
    prefix = np.empty(width, dtype=np.float64)
    for t in range(iterations):
        previous = history[t if return_history else t % 2]
        updated = history[t + 1 if return_history else (t + 1) % 2]
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
    if return_history:
        return history
    final = iterations % 2
    return history[final : final + 1]


@njit(cache=True, nogil=True)
def _decode_samples(initial, terms, iterations, mode, alpha, return_history):
    steps = iterations + 1 if return_history else 1
    result = np.empty((len(initial), steps, initial.shape[1]), dtype=np.float64)
    for sample in range(len(initial)):
        result[sample] = _iterate(initial[sample], terms, iterations, mode, alpha, return_history)
    return result


@njit(cache=True)
def _step_minsum(previous, updated, terms, alpha, scratch, counts):
    batch = previous.shape[1]
    minimum, second = scratch[0], scratch[1]
    negatives, zeros = counts[0], counts[1]
    for start in range(max(0, len(previous) - terms[-1])):
        minimum[:] = np.inf
        second[:] = np.inf
        negatives[:] = 0
        zeros[:] = 0
        for p in terms:
            for sample in range(batch):
                value = previous[start + p, sample]
                magnitude = abs(value)
                second[sample] = min(second[sample], max(minimum[sample], magnitude))
                minimum[sample] = min(minimum[sample], magnitude)
                negatives[sample] += value < 0
                zeros[sample] += value == 0
        for p in terms:
            for sample in range(batch):
                value = previous[start + p, sample]
                if zeros[sample] - (value == 0) > 0:
                    continue
                magnitude = second[sample] if abs(value) == minimum[sample] else minimum[sample]
                sign = -1.0 if (negatives[sample] - (value < 0)) % 2 else 1.0
                updated[start + p, sample] += alpha * sign * magnitude


@njit(cache=True)
def _step_sqrtsign(previous, updated, terms, prefix, scratch, counts):
    batch = previous.shape[1]
    width = len(terms)
    log_sum = scratch[0]
    negatives, zeros = counts[0], counts[1]
    # The same LLR participates in several checks: compute its logarithm once.
    for bit in range(len(previous)):
        for sample in range(batch):
            value = previous[bit, sample]
            prefix[bit, sample] = np.log(abs(value)) if value != 0 else 0.0
    for start in range(max(0, len(previous) - terms[-1])):
        log_sum[:] = 0
        negatives[:] = 0
        zeros[:] = 0
        for j in range(width):
            for sample in range(batch):
                value = previous[start + terms[j], sample]
                negatives[sample] += value < 0
                zeros[sample] += value == 0
                log_sum[sample] += prefix[start + terms[j], sample]
        for j in range(width):
            for sample in range(batch):
                value = previous[start + terms[j], sample]
                if zeros[sample] - (value == 0) > 0:
                    continue
                sign = -1.0 if (negatives[sample] - (value < 0)) % 2 else 1.0
                log_mean = (log_sum[sample] - prefix[start + terms[j], sample]) / (width - 1)
                updated[start + terms[j], sample] += sign * np.exp(log_mean)


@njit(cache=True)
def _step_tanh(previous, updated, terms, prefix, scratch):
    batch = previous.shape[1]
    width = len(terms)
    suffix = scratch[0]
    for start in range(max(0, len(previous) - terms[-1])):
        # Singleton XORs are copies: no infinity identity branches in these lanes.
        prefix[0] = previous[start + terms[0]]
        for j in range(1, width - 1):
            for sample in range(batch):
                prefix[j, sample] = _box_plus(
                    prefix[j - 1, sample], previous[start + terms[j], sample]
                )
        for sample in range(batch):
            updated[start + terms[-1], sample] += prefix[width - 2, sample]
        suffix[:] = previous[start + terms[-1]]
        for j in range(width - 2, 0, -1):
            for sample in range(batch):
                updated[start + terms[j], sample] += _box_plus(
                    prefix[j - 1, sample], suffix[sample]
                )
                suffix[sample] = _box_plus(previous[start + terms[j], sample], suffix[sample])
        for sample in range(batch):
            updated[start + terms[0], sample] += suffix[sample]


@njit(cache=True, nogil=True)
def _decode_batch(initial, terms, iterations, mode, alpha, return_history):
    batch, n = initial.shape
    if batch < 8:
        return _decode_samples(initial, terms, iterations, mode, alpha, return_history)
    # Store the batch axis contiguously: one check/tap traversal feeds all samples.
    previous = np.ascontiguousarray(initial.T)
    updated = np.empty_like(previous)
    prefix = np.empty((n if mode == 2 else len(terms), batch), dtype=np.float64)
    scratch = np.empty((2, batch), dtype=np.float64)
    counts = np.empty((2, batch), dtype=np.int64)
    steps = iterations + 1 if return_history else 1
    result = np.empty((batch, steps, n), dtype=np.float64)
    if return_history:
        result[:, 0, :] = initial
    for t in range(iterations):
        updated[:] = previous
        if mode == 1:
            _step_minsum(previous, updated, terms, alpha, scratch, counts)
        elif mode == 2:
            _step_sqrtsign(previous, updated, terms, prefix, scratch, counts)
        else:
            _step_tanh(previous, updated, terms, prefix, scratch)
        for bit in range(n):
            for sample in range(batch):
                if not np.isfinite(updated[bit, sample]):
                    raise FloatingPointError("LLR overflow: reduce T or input magnitudes")
        previous, updated = updated, previous
        if return_history:
            for sample in range(batch):
                for bit in range(n):
                    result[sample, t + 1, bit] = previous[bit, sample]
    if not return_history:
        for sample in range(batch):
            for bit in range(n):
                result[sample, 0, bit] = previous[bit, sample]
    return result


def decode(
    initial,
    terms=DEFAULT_TERMS,
    iterations=20,
    *,
    softxor: SoftXor = Tanh(),
    return_history=False,
) -> np.ndarray:
    """Decode a finite (B, N) LLR batch independently and synchronously.

    Return (B, N) final LLRs by default, or (B, T+1, N) with return_history=True.
    History includes the unchanged observation at t=0. Final-only decoding uses
    two rolling state buffers, never allocating the full history. For B >= 8,
    internal (N, B) storage keeps sample lanes contiguous and shares the check
    traversal across the batch. Smaller batches use the scalar path. Sqrt-sign
    reuses per-bit logarithms across all checks within an iteration.
    Time O(B*T*(N + max(0,N-K)*w))
    """
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    mode, alpha = _softxor_options(softxor)
    initial = np.asarray(initial, dtype=np.float64)
    if initial.ndim != 2 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("Initial LLRs must be a nonempty finite (B, N) batch")
    if not isinstance(return_history, (bool, np.bool_)):
        raise TypeError("return_history must be boolean")
    result = _decode_batch(
        np.ascontiguousarray(initial),
        np.asarray(terms, dtype=np.int64),
        iterations,
        mode,
        alpha,
        bool(return_history),
    )
    return result if return_history else result[:, 0, :]
