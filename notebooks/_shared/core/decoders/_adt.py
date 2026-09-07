"""Soft-XOR configuration ADT; each variant owns its applicable parameters."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Tanh:
    """Exact soft-XOR, evaluated with stable box-plus."""


@dataclass(frozen=True, slots=True)
class SqrtSign:
    """Product of signs times the geometric mean of magnitudes."""


@dataclass(frozen=True, slots=True)
class NormalizedMinSum:
    """Signed minimum magnitude, scaled once per opinion."""

    coefficient: float = 0.8

    def __post_init__(self):
        value = self.coefficient
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Normalized min-sum coefficient must be a real number")
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Normalized min-sum coefficient must be between 0 and 1")
        object.__setattr__(self, "coefficient", float(value))


SoftXor: TypeAlias = Tanh | NormalizedMinSum | SqrtSign
