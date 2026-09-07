"""Decoder architectures: decode(initial, terms, iterations, **options) -> history.

Each implementation returns a (T+1, N) ndarray with the observation at row zero.
Decoders never receive the true codeword or compute diagnostics/plots.
"""
