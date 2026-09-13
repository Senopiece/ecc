"""Architecture-independent diagnostics of a history against the true codeword."""

import numpy as np
from numba import njit

from ..core.lfsr import DEFAULT_TERMS, polynomial_terms


def validate_inputs(history, true):
    history = np.asarray(history, dtype=np.float64)
    true = np.asarray(true)
    if history.ndim != 2 or not all(history.shape) or not np.all(np.isfinite(history)):
        raise ValueError("History must be a nonempty finite (T+1, N) array")
    if true.ndim != 1 or len(true) != history.shape[1] or not np.all((true == 0) | (true == 1)):
        raise ValueError("True codeword must be a binary vector of length N")
    return history, np.asarray(true, dtype=np.uint8)


@njit(cache=True)
def _metrics(history, truth, terms):
    steps, n = history.shape
    count = max(0, n - terms[-1]) if len(terms) else 0
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


def metrics(history, true, terms=DEFAULT_TERMS):
    """Diagnostics; terms=None omits recurrence syndrome, e.g. for x histories."""
    history, true = validate_inputs(history, true)
    include_syndrome = terms is not None
    exponents = polynomial_terms(terms) if include_syndrome else ()
    values = _metrics(history, true, np.asarray(exponents, dtype=np.int64))
    names = (
        "Sign error rate (ties = 1/2)",
        "Erasure fraction",
        "Unsatisfied checks",
        "Mean logistic loss",
        "Mean signed margin",
    )
    return {
        name: value
        for i, (name, value) in enumerate(zip(names, values, strict=True))
        if include_syndrome or i != 2
    }
