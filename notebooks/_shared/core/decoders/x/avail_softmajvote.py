"""Recover x by additive column-to-row soft opinions through the generator graph."""

import numpy as np
from numba import njit

from ...generator import column_supports
from ...lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms
from .._adt import SoftXor, Tanh
from .._softxor import _box_plus, _softxor_options


@njit(cache=True, nogil=True)
def _decode(observed, initial, offsets, indices, iterations, mode, alpha, return_history):
    batch, k = initial.shape
    result = np.empty((batch, iterations + 1 if return_history else 1, k))
    for sample in range(batch):
        previous = initial[sample].copy()
        updated = np.empty(k)
        prefix = np.empty(k + 1)
        if return_history:
            result[sample, 0] = previous
        for t in range(iterations):
            updated[:] = previous
            for column in range(observed.shape[1]):
                left, right = offsets[column], offsets[column + 1]
                degree = right - left
                channel = observed[sample, column]
                if mode == 0:
                    prefix[0] = channel
                    for j in range(degree - 1):
                        prefix[j + 1] = _box_plus(prefix[j], previous[indices[left + j]])
                    suffix = np.inf
                    for j in range(degree - 1, -1, -1):
                        bit = indices[left + j]
                        updated[bit] += _box_plus(prefix[j], suffix)
                        if j:
                            suffix = _box_plus(previous[bit], suffix)
                else:
                    negatives = int(channel < 0)
                    zeros = int(channel == 0)
                    minimum, second = abs(channel), np.inf
                    log_sum = np.log(abs(channel)) if channel != 0 else 0.0
                    for j in range(degree):
                        value = previous[indices[left + j]]
                        negatives += value < 0
                        zeros += value == 0
                        if mode == 1:
                            second = min(second, max(minimum, abs(value)))
                            minimum = min(minimum, abs(value))
                        else:
                            prefix[j] = np.log(abs(value)) if value != 0 else 0.0
                            log_sum += prefix[j]
                    for j in range(degree):
                        bit = indices[left + j]
                        value = previous[bit]
                        if zeros - (value == 0) > 0:
                            continue
                        sign = -1.0 if (negatives - (value < 0)) % 2 else 1.0
                        if mode == 1:
                            magnitude = second if abs(value) == minimum else minimum
                            updated[bit] += alpha * sign * magnitude
                        else:
                            updated[bit] += sign * np.exp((log_sum - prefix[j]) / degree)
            for bit in range(k):
                if not np.isfinite(updated[bit]):
                    raise FloatingPointError("LLR overflow: reduce T or input magnitudes")
            previous, updated = updated, previous
            if return_history:
                result[sample, t + 1] = previous
        if not return_history:
            result[sample, 0] = previous
    return result


def decode(
    initial,
    terms=DEFAULT_TERMS,
    iterations=20,
    *,
    softxor: SoftXor = Tanh(),
    x_initial=None,
    init_scale=0.1,
    seed=None,
    return_history=False,
) -> np.ndarray:
    """Read fixed y observations (B,N), return x LLRs (B,K) or history (B,T+1,K).

    Each message is softxor(channel_j, x_neighbors_except_target). All messages
    use the previous x state. They are added to that state with no damping or
    extrinsic subtraction. Truth never enters decoding. Default x initialization
    is independent random signs with magnitude init_scale; x_initial overrides it.
    """
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    mode, alpha = _softxor_options(softxor)
    observed = np.asarray(initial, dtype=np.float64)
    if observed.ndim != 2 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("Observation must be a nonempty finite (B, N) batch")
    if not isinstance(return_history, (bool, np.bool_)):
        raise TypeError("return_history must be boolean")
    shape = (len(observed), terms[-1])
    if x_initial is None:
        if not np.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("init_scale must be finite and positive")
        x_initial = (1.0 - 2 * np.random.default_rng(seed).integers(0, 2, shape)) * init_scale
    x_initial = np.asarray(x_initial, dtype=np.float64)
    if x_initial.shape != shape or not np.isfinite(x_initial).all():
        raise ValueError("x_initial must be a finite (B, K) batch")
    offsets, indices = column_supports(observed.shape[1], terms)
    result = _decode(
        np.ascontiguousarray(observed),
        np.ascontiguousarray(x_initial),
        offsets,
        indices,
        iterations,
        mode,
        alpha,
        bool(return_history),
    )
    return result if return_history else result[:, 0, :]
