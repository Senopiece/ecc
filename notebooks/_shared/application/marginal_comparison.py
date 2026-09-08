"""Paired AWGN experiments, process-pool execution, and compact sample statistics."""

from time import perf_counter_ns

import numpy as np
from joblib import Parallel, cpu_count, delayed, parallel_config
from tqdm.auto import tqdm

from ..core.channel import sample_batch
from ..core.lfsr import integer_parameter, polynomial_terms


def _warm_decoders(decoders, terms, n):
    # Runs once per worker, outside any reported timing. Also warm the encoder.
    initial, _, _ = sample_batch(np.random.default_rng(0), 1, n, terms, (-1, 1))
    for decoder in decoders.values():
        decoder(initial.copy(), terms=terms, return_history=False)


def _batch_task(index, seed, decoders, batch_size, n, terms, snr_db_range):
    initial, truth, snr_db = sample_batch(
        np.random.default_rng(seed), batch_size, n, terms, snr_db_range
    )
    statistics = {}
    names = list(decoders)
    # Rotate execution order to reduce systematic first/last-decoder effects.
    names = names[index % len(names) :] + names[: index % len(names)]
    for name in names:
        private = initial.copy()
        started = perf_counter_ns()
        final = decoders[name](private, terms=terms, return_history=False)
        elapsed = perf_counter_ns() - started
        if not isinstance(final, np.ndarray) or final.shape != initial.shape:
            raise ValueError(f"{name}: expected a (B, N) final LLR array")
        if not np.isfinite(final).all():
            raise ValueError(f"{name}: nonfinite decoded LLRs")
        wrong = ((final < 0) != truth) & (final != 0)
        ser = (wrong.sum(axis=1) + 0.5 * (final == 0).sum(axis=1)) / n
        statistics[name] = dict(decode_ns=elapsed, ser=ser)
    return index, snr_db, statistics


def run_experiment(
    decoders,
    *,
    terms,
    n=32,
    snr_db_range=(-3, 3),
    batch_size=128,
    batches=1024,
    workers=None,
    seed=0,
):
    """One task = generate one batch, decode with all methods, return statistics.

    Per-batch seeds and ordered assembly make observations independent of worker
    count and task completion order. Decoder callables bind their own iteration
    counts. Timings exclude JIT warmup, generation, copying, IPC, and statistics;
    they include public decode validation/output allocation under CPU contention.
    """
    if not decoders or any(not callable(d) for d in decoders.values()):
        raise ValueError("Supply a nonempty mapping of names to decoder callables")
    terms = polynomial_terms(terms)
    n = integer_parameter(n, "N", 1)
    batch_size = integer_parameter(batch_size, "batch_size", 1)
    batches = integer_parameter(batches, "batches", 1)
    workers = cpu_count() if workers is None else integer_parameter(workers, "workers", 1)
    workers = min(workers, batches)
    # Validate channel parameters before starting worker processes.
    sample_batch(np.random.default_rng(0), 1, n, terms, snr_db_range)
    seeds = np.random.SeedSequence(seed).spawn(batches)
    results = {
        name: dict(ser=np.empty((batches, batch_size)), decode_ns=np.empty(batches, np.int64))
        for name in decoders
    }
    snr_db = np.empty((batches, batch_size))
    if workers == 1:
        _warm_decoders(decoders, terms, n)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=workers,
            return_as="generator_unordered",
            batch_size=1,
            initializer=_warm_decoders,
            initargs=(decoders, terms, n),
        ) as pool:
            completed = pool(
                delayed(_batch_task)(i, s, decoders, batch_size, n, terms, snr_db_range)
                for i, s in enumerate(seeds)
            )
            for index, batch_snr, stats in tqdm(
                completed, total=batches, desc="Batches", unit="batch"
            ):
                snr_db[index] = batch_snr
                for name, values in stats.items():
                    results[name]["ser"][index] = values["ser"]
                    results[name]["decode_ns"][index] = values["decode_ns"]
    for values in results.values():
        values["ser"] = values["ser"].ravel()
        values["ns_per_sample"] = values["decode_ns"].sum() / (batches * batch_size)
    return dict(snr_db=snr_db.ravel(), decoders=results, snr_db_range=tuple(snr_db_range), n=n)


def conditional_ser(snr_db, ser, snr_db_range, bins=30, ser_bins=64):
    """Histogram probability P(SER cell | SNR cell); each populated column sums to 1."""
    bins = integer_parameter(bins, "bins", 1)
    ser_bins = integer_parameter(ser_bins, "ser_bins", 1)
    snr_db, ser = np.asarray(snr_db), np.asarray(ser)
    if snr_db.ndim != 1 or ser.shape != snr_db.shape:
        raise ValueError("SNR and SER must be matching vectors")
    low, high = snr_db_range
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("SNR range must be finite and increasing")
    if (
        not np.isfinite(snr_db).all()
        or not np.isfinite(ser).all()
        or np.any((ser < 0) | (ser > 1))
        or np.any((snr_db < low) | (snr_db > high))
    ):
        raise ValueError("Samples must be finite and within the SNR range and SER [0, 1]")
    xedges = np.linspace(low, high, bins + 1)
    yedges = np.linspace(0, 1, ser_bins + 1)
    counts = np.histogram2d(snr_db, ser, bins=(xedges, yedges))[0].T
    column_count = counts.sum(axis=0)
    probability = np.full_like(counts, np.nan)
    np.divide(counts, column_count[None, :], out=probability, where=column_count[None, :] > 0)
    sums = np.histogram(snr_db, bins=xedges, weights=ser)[0]
    mean = np.full(bins, np.nan)
    np.divide(sums, column_count, out=mean, where=column_count > 0)
    return dict(
        snr=(xedges[:-1] + xedges[1:]) / 2,
        snr_edges=xedges,
        ser_edges=yedges,
        mean=mean,
        probability=probability,
        cell_count=counts,
        count=column_count,
    )
