"""Decoder architectures: decode(initial_batch, terms, iterations, **options).

Input is observed (B, N) LLRs. Modules under y return (B, N) codeword LLRs;
modules under x return (B, K) information-bit LLRs. return_history=True
adds a T+1 axis after the batch axis, with the initial state at iteration zero.
Decoders never receive the true codeword or compute diagnostics/plots.
"""
