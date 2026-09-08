"""BPSK over real AWGN with SNR defined as Es/N0 and Es=1."""

import numpy as np
from numba import njit

from .lfsr import _encode, integer_parameter, polynomial_terms


@njit(cache=True)
def _encode_batch(bits, n, taps, k):
    result = np.empty((len(bits), n), dtype=np.uint8)
    for sample in range(len(bits)):
        result[sample] = _encode(bits[sample], n, taps, k)
    return result


def sample_batch(rng, batch_size, n, terms, snr_db_range):
    """Return (LLR[B,N], true_bits[B,N], SNR_dB[B]). Each sample has its own SNR."""
    terms = polynomial_terms(terms)
    batch_size = integer_parameter(batch_size, "batch_size", 1)
    n = integer_parameter(n, "N", 1)
    low, high = snr_db_range
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("SNR range must contain two finite, increasing endpoints")
    bits = rng.integers(0, 2, (batch_size, terms[-1]), dtype=np.uint8)
    snr_db = rng.uniform(low, high, batch_size)
    variance = 1 / (2 * 10 ** (snr_db / 10))
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("SNR range produces an unrepresentable noise variance")
    truth = _encode_batch(bits, n, np.asarray(terms[:-1], dtype=np.int64), terms[-1])
    received = 1.0 - 2.0 * truth + rng.normal(size=truth.shape) * np.sqrt(variance[:, None])
    llr = 2 * received / variance[:, None]
    return llr, truth, snr_db
