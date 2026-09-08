"""Decoder architectures: decode(initial_batch, terms, iterations, **options).

Input is (B, N). Return final (B, N) LLRs by default; return_history=True selects
a (B, T+1, N) ndarray with the observation at iteration zero.
Decoders never receive the true codeword or compute diagnostics/plots.
"""
