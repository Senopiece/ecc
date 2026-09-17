"""Unknown-prefix-invariant PR superposition with a known nonzero phase block."""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil

import numpy as np
from numba import njit

from ..decoders.x.osd import OSDConfig, decode_packed, patterns
from ..finite_fields import BinaryField, choose_degree, field
from ..lfsr import integer_parameter


@njit(cache=True)
def _prefix_masks(degree, exponents, length, exp, log, trace):
    period = len(exp)
    masks = np.zeros((len(exponents), length), dtype=np.uint32)
    for carrier, d in enumerate(exponents):
        for bit in range(degree):
            exponent = log[1 << bit]
            for t in range(length):
                masks[carrier, t] |= np.uint32(trace[exp[exponent]]) << np.uint32(bit)
                exponent = (exponent + d) % period
    return masks


@njit(cache=True)
def _pack_generator(masks, degree):
    components, length = masks.shape
    packed = np.zeros((components * degree, (length + 63) // 64), dtype=np.uint64)
    for component in range(components):
        for bit in range(degree):
            for column in range(length):
                if (masks[component, column] >> bit) & 1:
                    packed[component * degree + bit, column >> 6] |= np.uint64(1) << np.uint64(
                        column & 63
                    )
    return packed


@njit(cache=True)
def encode_window(message, degree, masks, parity, shift, length):
    """Only evaluate the requested [shift, shift+length) output, never the lost prefix."""
    components = (len(message) + degree - 1) // degree + 1
    blocks = np.zeros(components, dtype=np.int64)
    for i in range(len(message)):
        blocks[i // degree] |= int(message[i]) << (i % degree)
    blocks[-1] = 1  # Field element 1, not an all-ones bit vector.
    y = np.zeros(length, dtype=np.uint8)
    for t in range(length):
        for component in range(components):
            y[t] ^= parity[blocks[component] & masks[component, shift + t]]
    return y


@njit(cache=True)
def repair_phase(packed, message_bits, degree, exponents, pilot_inverse, exp, log):
    """No padding inspection. Return (original K bits, nonzero pilot)."""
    components = len(exponents)
    transformed = np.zeros(components, dtype=np.int64)
    for i in range(components * degree):
        bit = (packed[i >> 6] >> np.uint64(i & 63)) & np.uint64(1)
        transformed[i // degree] |= int(bit) << (i % degree)
    result = np.zeros(message_bits, dtype=np.uint8)
    pilot = transformed[-1]
    if pilot == 0:
        return result, False
    period = len(exp)
    # The phase block is on carrier alpha^d_p, so invert d_p modulo the period.
    phase = (log[pilot] * pilot_inverse) % period
    for component in range(components - 1):
        value = transformed[component]
        if value:
            value = exp[(log[value] - phase * exponents[component]) % period]
        for bit in range(degree):
            index = component * degree + bit
            if index < message_bits:
                result[index] = (value >> bit) & 1
    return result, True


@njit(cache=True, nogil=True)
def simulate_batch(
    messages,
    lengths,
    shifts,
    snr_db,
    noise,
    degree,
    masks,
    parity,
    generator_rows,
    error_patterns,
    exponents,
    pilot_inverse,
    exp,
    log,
):
    successes = np.zeros(len(messages), dtype=np.bool_)
    valid = np.zeros(len(messages), dtype=np.bool_)
    for trial in range(len(messages)):
        length = lengths[trial]
        y = encode_window(messages[trial], degree, masks, parity, shifts[trial], length)
        variance = 1 / (2 * 10 ** (snr_db[trial] / 10))
        llr = np.empty(length)
        for i in range(length):
            llr[i] = 2 * (1.0 - 2.0 * y[i] + np.sqrt(variance) * noise[trial, i]) / variance
        packed, rank_ok = decode_packed(llr, generator_rows, error_patterns)
        if rank_ok:
            decoded, phase_ok = repair_phase(
                packed, messages.shape[1], degree, exponents, pilot_inverse, exp, log
            )
            valid[trial] = phase_ok
            successes[trial] = phase_ok and np.all(decoded == messages[trial])
    return successes, valid


@dataclass(frozen=True)
class PreparedShiftPR:
    message_bits: int
    degree: int
    padding: int
    max_shift: int
    max_length: int
    binary_field: BinaryField
    exponents: np.ndarray
    pilot_inverse: int
    masks: np.ndarray
    generator_rows: np.ndarray
    error_patterns: np.ndarray

    @property
    def dimension(self):
        return self.message_bits + self.padding + self.degree

    def encode(self, message, *, length, shift=0):
        message = np.asarray(message)
        if message.shape != (self.message_bits,) or not np.isin(message, [0, 1]).all():
            raise ValueError("Expected K binary information bits")
        length = integer_parameter(length, "N", 1)
        shift = integer_parameter(shift, "S")
        if length > self.max_length or shift > self.max_shift:
            raise ValueError("Requested window exceeds the prepared prefix")
        return encode_window(
            np.asarray(message, np.uint8),
            self.degree,
            self.masks,
            self.binary_field.parity,
            shift,
            length,
        )

    def decode(self, llr):
        """The decoder is never given S, original bits, or padding constraints."""
        llr = np.asarray(llr, dtype=np.float64)
        if (
            llr.ndim != 1
            or not len(llr)
            or len(llr) > self.max_length
            or not np.isfinite(llr).all()
        ):
            raise ValueError("Invalid LLR observation")
        packed, valid = decode_packed(llr, self.generator_rows, self.error_patterns)
        if not valid:
            return None
        result, valid = repair_phase(
            packed,
            self.message_bits,
            self.degree,
            self.exponents,
            self.pilot_inverse,
            self.binary_field.exp,
            self.binary_field.log,
        )
        return result if valid else None

    def simulate(self, messages, lengths, shifts, snr_db, noise):
        f = self.binary_field
        return simulate_batch(
            messages,
            lengths,
            shifts,
            snr_db,
            noise,
            self.degree,
            self.masks,
            f.parity,
            self.generator_rows,
            self.error_patterns,
            self.exponents,
            self.pilot_inverse,
            f.exp,
            f.log,
        )


@lru_cache(maxsize=16)
def _family_banks(message_sizes, c, max_shift, max_overhead):
    """One prefix per degree shared across K values and decoder configurations."""
    requirements = {}
    for size in message_sizes:
        degree, padding, _ = choose_degree(size, c)
        components = (size + padding) // degree + 1
        length = ceil((1 + max_overhead) * (size + padding + degree))
        old = requirements.get(degree, (0, 0))
        requirements[degree] = (max(old[0], components), max(old[1], length))
    banks = {}
    for degree, (components, length) in requirements.items():
        f = field(degree)
        if components > len(f.carriers):
            raise ValueError("Not enough inequivalent primitive carriers at the selected degree")
        masks = _prefix_masks(
            degree, f.carriers[:components], max_shift + length, f.exp, f.log, f.trace
        )
        generator = _pack_generator(masks[:, :length], degree)
        masks.setflags(write=False)
        generator.setflags(write=False)
        banks[degree] = (f, masks, generator)
    return banks


@dataclass(frozen=True)
class ShiftInvariantPR:
    name: str = "shift-invariant-pr/osd"
    c: int = 32
    osd: OSDConfig = OSDConfig()

    def prepare(self, message_sizes, *, max_shift, max_overhead):
        sizes = tuple(sorted(set(integer_parameter(k, "K", 1) for k in message_sizes)))
        integer_parameter(self.c, "C", 1)
        integer_parameter(max_shift, "S_max")
        if not np.isfinite(max_overhead) or max_overhead < 0:
            raise ValueError("max_overhead must be finite and nonnegative")
        banks = _family_banks(sizes, self.c, max_shift, float(max_overhead))
        result = {}
        for size in sizes:
            degree, padding, _ = choose_degree(size, self.c)
            dimension = size + padding + degree
            components = dimension // degree
            f, masks, generator = banks[degree]
            exponents = f.carriers[:components]
            result[size] = PreparedShiftPR(
                size,
                degree,
                padding,
                max_shift,
                ceil((1 + max_overhead) * dimension),
                f,
                exponents,
                pow(int(exponents[-1]), -1, len(f.exp)),
                masks[:components],
                generator[:dimension],
                patterns(dimension, self.osd),
            )
        return result
