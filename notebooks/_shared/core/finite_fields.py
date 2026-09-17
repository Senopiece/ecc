"""Small binary fields and complete nested families of primitive carriers."""

from dataclasses import dataclass
from functools import lru_cache
from math import log2

import numpy as np
from numba import njit

from .lfsr import integer_parameter


def choose_degree(message_bits, c=32):
    """Three nearest integers to log2(C*K), then minimize (padding+k, padding, k)."""
    message_bits = integer_parameter(message_bits, "K", 1)
    c = integer_parameter(c, "C", 1)
    target = log2(c * message_bits)
    center = int(target)
    candidates = sorted(
        range(max(2, center - 2), max(5, center + 4)), key=lambda k: (abs(k - target), k)
    )[:3]
    degree = min(candidates, key=lambda k: ((-message_bits) % k + k, (-message_bits) % k, k))
    return degree, (-message_bits) % degree, tuple(sorted(candidates))


def prime_factors(value):
    result = []
    p = 2
    while p * p <= value:
        if value % p == 0:
            result.append(p)
            while value % p == 0:
                value //= p
        p += 1
    if value > 1:
        result.append(value)
    return tuple(result)


@njit(cache=True)
def multiply(a, b, modulus, degree):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & (1 << degree):
            a ^= modulus
    return result


@njit(cache=True)
def power(a, exponent, modulus, degree):
    result = 1
    while exponent:
        if exponent & 1:
            result = multiply(result, a, modulus, degree)
        a = multiply(a, a, modulus, degree)
        exponent >>= 1
    return result


@njit(cache=True)
def _poly_degree(a):
    result = -1
    while a:
        a >>= 1
        result += 1
    return result


@njit(cache=True)
def _poly_gcd(a, b):
    while b:
        while a and _poly_degree(a) >= _poly_degree(b):
            a ^= b << (_poly_degree(a) - _poly_degree(b))
        a, b = b, a
    return a


@njit(cache=True)
def _primitive(modulus, degree, factors):
    # Rabin irreducibility test, followed by the multiplicative-order test.
    x = 2
    for i in range(1, degree + 1):
        x = multiply(x, x, modulus, degree)
        if i <= degree // 2 and _poly_gcd(x ^ 2, modulus) != 1:
            return False
    if x != 2:
        return False
    period = (1 << degree) - 1
    for p in factors:
        if power(2, period // p, modulus, degree) == 1:
            return False
    return True


@njit(cache=True)
def _primitive_modulus(degree, factors):
    """Find one field modulus; carrier families are generated from root exponents."""
    for modulus in range((1 << degree) | 1, 1 << (degree + 1), 2):
        if _primitive(modulus, degree, factors):
            return modulus
    raise ValueError("No primitive modulus found")


@njit(cache=True)
def _cosets_sieve(degree, factors):
    period = (1 << degree) - 1
    eligible = np.ones(period, dtype=np.bool_)
    for p in factors:
        eligible[::p] = False
    eligible[0] = False
    visited = np.zeros(period, dtype=np.bool_)
    result = []
    for d in range(1, period):
        if eligible[d] and not visited[d]:
            result.append(d)
            e = d
            for _ in range(degree):
                visited[e] = True
                e = (2 * e) % period
    return np.asarray(result, dtype=np.int64)


@njit(cache=True)
def _tables(degree, modulus):
    period = (1 << degree) - 1
    exp = np.empty(period, dtype=np.int64)
    log = np.full(period + 1, -1, dtype=np.int64)
    value = 1
    for i in range(period):
        exp[i], log[value] = value, i
        value <<= 1
        if value & (1 << degree):
            value ^= modulus
    trace = np.zeros(period + 1, dtype=np.uint8)
    parity = np.zeros(period + 1, dtype=np.uint8)
    for value in range(1, period + 1):
        exponent, total = log[value], 0
        for _ in range(degree):
            total ^= exp[exponent]
            exponent = (2 * exponent) % period
        trace[value] = total
        parity[value] = parity[value >> 1] ^ (value & 1)
    return exp, log, trace, parity


@dataclass(frozen=True)
class BinaryField:
    degree: int
    modulus: int
    exp: np.ndarray
    log: np.ndarray
    trace: np.ndarray
    parity: np.ndarray
    carriers: np.ndarray


@lru_cache(maxsize=16)
def field(degree):
    degree = integer_parameter(degree, "degree", 2)
    if degree > 20:
        raise ValueError("Table-based experiments currently support degrees 2..20")
    factors = np.asarray(prime_factors((1 << degree) - 1), dtype=np.int64)
    modulus = int(_primitive_modulus(degree, factors))
    exp, log, trace, parity = _tables(degree, modulus)
    carriers = _cosets_sieve(degree, factors)
    for values in (exp, log, trace, parity, carriers):
        values.setflags(write=False)
    return BinaryField(degree, modulus, exp, log, trace, parity, carriers)
