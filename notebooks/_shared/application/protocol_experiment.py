"""Reproducible protocol trials with shared prepared assets and in-memory results."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from tqdm.auto import tqdm

from ..core.lfsr import integer_parameter


@dataclass(frozen=True)
class ExperimentConfig:
    experiments: int = 1_000_000
    message_sizes: tuple = (32, 64, 128, 256)
    snr_db_range: tuple = (-3.0, 3.0)
    overhead_range: tuple = (0.1, 0.5)
    shift_range: tuple = (0, 4095)  # Inclusive; recovery only needs S modulo the period.
    batch_size: int = 256
    seed: int = 42

    def __post_init__(self):
        for name in ("experiments", "batch_size"):
            integer_parameter(getattr(self, name), name, 1)
        integer_parameter(self.seed, "seed")
        if not self.message_sizes or len(set(self.message_sizes)) != len(self.message_sizes):
            raise ValueError("message_sizes must be nonempty and distinct")
        for size in self.message_sizes:
            integer_parameter(size, "K", 1)
        for name in ("snr_db_range", "overhead_range", "shift_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid {name}")
        if self.overhead_range[0] < 0:
            raise ValueError("Overhead must be nonnegative")
        for shift in self.shift_range:
            integer_parameter(shift, "shift")


def _preparation_signature(protocols, config):
    return (
        tuple(repr(p) for p in protocols),
        tuple(config.message_sizes),
        config.shift_range[1],
        config.overhead_range[1],
    )


class PreparedProtocols(dict):
    """Mapping with the preparation settings needed to reject stale notebook state."""

    def __init__(self, entries, signature):
        super().__init__(entries)
        self.signature = signature


def prepare_protocols(protocols, config):
    """A protocol supplies name/prepare; prepared instances supply dimension/simulate.

    Each spec's prepare returns {K: prepared}. A prepared instance exposes
    dimension and max_length, and simulate(messages, lengths, shifts, snr_db,
    standard_normal_noise) -> (success, valid) boolean arrays. Numerical work
    belongs to core; this module only samples parameters and assembles tables.
    """
    protocols = tuple(protocols)
    if not protocols or len({p.name for p in protocols}) != len(protocols):
        raise ValueError("Supply protocols with distinct names")
    entries = {
        p.name: p.prepare(
            config.message_sizes,
            max_shift=config.shift_range[1],
            max_overhead=config.overhead_range[1],
        )
        for p in protocols
    }
    return PreparedProtocols(entries, _preparation_signature(protocols, config))


def warmup_protocols(prepared):
    """Explicit compilation pass, outside experiment timing and random streams."""
    for variants in prepared.values():
        for size, asset in variants.items():
            asset.simulate(
                np.zeros((1, size), dtype=np.uint8),
                np.array([asset.max_length], dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1),
                np.zeros((1, asset.max_length)),
            )


def _chunk(index, prepared, config):
    start = index * config.batch_size
    count = min(config.batch_size, config.experiments - start)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, index]))
    names = tuple(prepared)
    proto = rng.integers(len(names), size=count)
    sizes = rng.choice(config.message_sizes, count)
    overhead = rng.uniform(*config.overhead_range, count)
    snr = rng.uniform(*config.snr_db_range, count)
    shifts = rng.integers(config.shift_range[0], config.shift_range[1] + 1, count)
    lengths = np.empty(count, np.int64)
    dimensions = np.empty(count, np.int64)
    success = np.empty(count, bool)
    valid = np.empty(count, bool)
    for p, name in enumerate(names):
        for size, asset in prepared[name].items():
            rows = np.flatnonzero((proto == p) & (sizes == size))
            if not len(rows):
                continue
            dimensions[rows] = asset.dimension
            lengths[rows] = np.ceil((1 + overhead[rows]) * asset.dimension).astype(np.int64)
            messages = rng.integers(0, 2, (len(rows), size), dtype=np.uint8)
            noise = rng.standard_normal((len(rows), int(lengths[rows].max())))
            success[rows], valid[rows] = asset.simulate(
                messages, lengths[rows], shifts[rows], snr[rows], noise
            )
    return index, pd.DataFrame(
        {
            "Trial": np.arange(start, start + count),
            "Proto": np.asarray(names)[proto],
            "K": sizes,
            "Overhead": overhead,
            "SNR": snr,
            "S": shifts,
            "N": lengths,
            "D": dimensions,
            "RealizedOverhead": lengths / dimensions - 1,
            "succ": success,
            "valid": valid,
        }
    )


def run_experiment(protocols, config, *, prepared=None, workers=None, progress=True):
    """CPU threads share immutable banks; compiled batches release the GIL.

    Compilation must be done with warmup_protocols before calling this function.
    Per-chunk seeds make results independent of worker count and completion order.
    Every invocation runs all trials and returns an in-memory table.
    """
    protocols = tuple(protocols)
    prepared = prepare_protocols(protocols, config) if prepared is None else prepared
    if getattr(prepared, "signature", None) != _preparation_signature(protocols, config):
        raise ValueError("Prepared assets do not match the settings; rerun the preparation cell")
    workers = cpu_count() if workers is None else integer_parameter(workers, "workers", 1)
    chunks = (config.experiments + config.batch_size - 1) // config.batch_size
    results = [None] * chunks
    started = perf_counter()
    with Parallel(
        n_jobs=min(workers, chunks),
        backend="threading",
        batch_size=1,
        return_as="generator_unordered",
    ) as pool:
        completed = pool(delayed(_chunk)(i, prepared, config) for i in range(chunks))
        for index, frame in tqdm(
            completed, total=chunks, desc="Protocol batches", disable=not progress
        ):
            results[index] = frame
    table = pd.concat(results, ignore_index=True)
    table["Proto"] = table["Proto"].astype("category")
    table.attrs["elapsed_seconds"] = perf_counter() - started
    return table


def success_slices(table, *, axes, filter_axis, edges):
    """Pool counts for ALL or each third-axis bin; never average probabilities."""
    columns = (*axes, filter_axis)
    coordinates = table[list(columns)].to_numpy()
    bins = [edges[column] for column in columns]
    count, _ = np.histogramdd(coordinates, bins=bins)
    successful, _ = np.histogramdd(
        coordinates, bins=bins, weights=table["succ"].to_numpy(dtype=float)
    )
    slices = [(successful.sum(axis=2), count.sum(axis=2))]
    slices.extend((successful[:, :, i], count[:, :, i]) for i in range(count.shape[2]))
    result = []
    for numerator, denominator in slices:
        probability = np.full_like(denominator, np.nan)
        np.divide(numerator, denominator, out=probability, where=denominator > 0)
        result.append((probability.T, denominator.T))
    return result
