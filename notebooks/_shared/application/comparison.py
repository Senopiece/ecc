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
        batch_history = decoder(
            initial[None, :].copy(), terms=terms, iterations=iterations, return_history=True
        )
        if not isinstance(batch_history, np.ndarray) or batch_history.shape != (
            1,
            iterations + 1,
            initial.size,
        ):
            raise ValueError(f"{name}: decode must return a (1, T+1, N) history")
        history = batch_history[0]
        if not isinstance(history, np.ndarray) or history.shape != (iterations + 1, initial.size):
            raise ValueError(f"{name}: decode must return a (T+1, N) ndarray")
        if not np.all(np.isfinite(history)) or not np.array_equal(history[0], initial):
            raise ValueError(f"{name}: history must be finite and start with the observation")
        histories[name] = history
    return histories


def run_x_decoders(decoders, observed, x_initial, terms=DEFAULT_TERMS, iterations=20):
    """Compare recovery of x with shared initial beliefs and fixed y observations."""
    terms = polynomial_terms(terms)
    iterations = integer_parameter(iterations, "T")
    observed = np.asarray(observed, dtype=np.float64)
    x_initial = np.asarray(x_initial, dtype=np.float64)
    if observed.ndim != 1 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("Observation must be a nonempty finite vector")
    if x_initial.shape != (terms[-1],) or not np.isfinite(x_initial).all():
        raise ValueError("x_initial must contain K finite LLRs")
    if not decoders:
        raise ValueError("Select at least one decoder")
    histories = {}
    for name, decoder in decoders.items():
        if not isinstance(name, str) or not name.strip() or not callable(decoder):
            raise ValueError("Each decoder needs a name and a callable")
        result = decoder(
            observed[None, :].copy(),
            terms=terms,
            iterations=iterations,
            x_initial=x_initial[None, :].copy(),
            return_history=True,
        )
        if not isinstance(result, np.ndarray) or result.shape != (
            1,
            iterations + 1,
            len(x_initial),
        ):
            raise ValueError(f"{name}: expected a (1, T+1, K) history")
        if not np.isfinite(result).all() or not np.array_equal(result[0, 0], x_initial):
            raise ValueError(f"{name}: invalid history or initial state")
        histories[name] = result[0]
    return histories
