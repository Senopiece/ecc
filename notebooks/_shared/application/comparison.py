"""Run an ordered collection of decoder callables on the same observation."""

import numpy as np

from ..core.lfsr import DEFAULT_TERMS, integer_parameter, polynomial_terms


def run_decoders(decoders, initial, terms=DEFAULT_TERMS, iterations=20):
    """Return {decoder_name: history}; no true bits, metrics or plots are involved.

    Extra decoder parameters are bound by the caller, e.g. functools.partial.
    Dict insertion order determines decoder order in plot selectors.
    Each decoder gets a private input copy, even if it modifies its input.
    """
    if not decoders:
        raise ValueError("Select at least one decoder")
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    initial = np.array(initial, dtype=np.float64, copy=True)
    if initial.ndim != 1 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("Observation must be a nonempty finite vector")
    histories = {}
    for name, decoder in decoders.items():
        if not isinstance(name, str) or not name.strip() or not callable(decoder):
            raise ValueError("Each decoder entry needs a name and a callable")
        history = decoder(initial.copy(), terms=terms, iterations=iterations)
        if not isinstance(history, np.ndarray) or history.shape != (iterations + 1, initial.size):
            raise ValueError(f"{name}: decode must return a (T+1, N) ndarray")
        if not np.all(np.isfinite(history)) or not np.array_equal(history[0], initial):
            raise ValueError(f"{name}: history must be finite and start with the observation")
        histories[name] = history
    return histories
