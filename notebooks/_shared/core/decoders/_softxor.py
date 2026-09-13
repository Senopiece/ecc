"""Shared soft-XOR mode dispatch and stable binary box-plus."""

import numpy as np
from numba import njit

from . import _adt


def _softxor_options(softxor: _adt.SoftXor):
    """Lower the configuration to scalar arguments for the compiled kernel."""
    match softxor:
        case _adt.Tanh():
            return 0, 1.0
        case _adt.NormalizedMinSum(coefficient=alpha):
            return 1, alpha
        case _adt.SqrtSign():
            return 2, 1.0
        case _:
            raise TypeError("softxor must be Tanh(), NormalizedMinSum(...), or SqrtSign()")


@njit(cache=True, inline="always")
def _box_plus(a, b):
    # +infinity is the identity for the empty prefix/suffix of an XOR.
    if a == np.inf:
        return b
    if b == np.inf:
        return a
    if a == 0.0 or b == 0.0:
        return 0.0
    magnitude = min(abs(a), abs(b))
    sign = 1.0 if (a > 0) == (b > 0) else -1.0
    return sign * magnitude + np.log1p(np.exp(-abs(a + b))) - np.log1p(np.exp(-abs(a - b)))
