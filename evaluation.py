from __future__ import annotations

import csv
import ctypes
import random
import threading
import timeit
from multiprocessing import Pool, TimeoutError, cpu_count
from pathlib import Path

from gap_aware_alignments import ReachabilityGraph, align
from gap_aware_alignments.petri_net import PetriNet

# Set seed for reproducibility
random.seed(42)
# Number of worker processes to use
N_CPUS = cpu_count()
N_WORKERS = max(1, min(N_CPUS - 2, 61))
TIMEOUT = 300
MAX_TRACE_VARIANTS = 1000
OFFSET = 0


class _AlignmentTimeout(Exception):
    pass


def _interrupt_thread(thread_id: int) -> None:
    """Inject _AlignmentTimeout into a running thread via the CPython C-API."""
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id),
        ctypes.py_object(_AlignmentTimeout),
    )


def _timed_call(fn, timeout: float):
    """Run ``fn()`` in a daemon thread and measure wall time with ``timeit``.

    If ``fn()`` does not complete within ``timeout`` seconds, ``_AlignmentTimeout``
    is injected into the thread and re-raised here.  Returns ``(result, elapsed)``.
    """
    outcome = []  # filled as [result, elapsed] by the thread
    error = [None]

    def _target():
        try:
            elapsed = timeit.timeit(lambda: outcome.append(fn()), number=1)
            outcome.append(elapsed)  # outcome == [result, elapsed]
        except _AlignmentTimeout:
            pass
        except Exception as e:  # noqa: BLE001
            error[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        _interrupt_thread(thread.ident)
        thread.join(timeout=5)  # grace period for cleanup
        raise _AlignmentTimeout()

    if error[0] is not None:
        raise error[0]

    if len(outcome) < 2:
        raise _AlignmentTimeout()

    return outcome[0], outcome[1]


def evaluate_trace_gap_aware(
    trace: tuple[str, ...],
    reachability_graph: ReachabilityGraph,
    gap_opening_cost: float = 1.0,
    gap_extension_cost: float = 0.5,
    gap_power: float = 0.7,
    repeat: int = 10,
) -> list[tuple]:
    runs = []
    for _ in range(repeat):
        try:
            result, elapsed = _timed_call(
                lambda: align(
                    trace,
                    reachability_graph,
                    gap_opening_cost=gap_opening_cost,
                    gap_extension_cost=gap_extension_cost,
                    gap_power=gap_power,
                ),
                TIMEOUT,
            )
            runs.append((result, elapsed))
        except _AlignmentTimeout:
            break
    return runs


def _run_block(pool: Pool, fn, args_iter, output_path: Path, repeat: int = 10) -> None:
    """Submit one task per variant, collect best-of-repeat results, write to CSV."""
    pending = [
        (variant, freq, pool.apply_async(fn, args=args + (repeat,)))
        for variant, freq, args in args_iter
    ]
    total = len(pending)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for done, (variant, freq, async_result) in enumerate(pending, start=1):
            try:
                runs = async_result.get(timeout=(repeat + 1) * TIMEOUT)
                if not runs:
                    raise TimeoutError
                # runs is a list of ((costs_tuple, alignments_tuple), elapsed)
                cost_results = [r for r, _ in runs]
                costs = [c for c, _ in cost_results]
                alignments = [a for _, a in cost_results]
                times = [t for _, t in runs]
                if len(set(costs)) > 1:
                    raise ValueError(f"Cost mismatch across repetitions for variant {variant}: {costs}")
                a = alignments[0]
                writer.writerow(
                    [
                        costs[0],
                        min(times),
                        freq,
                        len(variant),
                        variant,
                        a[0]["alignment"],
                        a[1]["alignment"],
                        a[2]["alignment"],
                        a[3]["alignment"],
                    ]
                )
            except TimeoutError:
                writer.writerow(["timeout", "timeout", freq, len(variant), variant] + ["timeout"] * 4)
            finally:
                f.flush()
            if done == total or done % 25 == 0:
                print(f"  [{done}/{total}] variants aligned", flush=True)


def evaluate_event_log(
    trace_variants: list[tuple[tuple[str, ...], int]],
    accepting_petri_net: PetriNet,
    repeat: int = 5,
    result_path: str | Path = "output",
    file_tag: str = "",
    max_trace_variants: int = MAX_TRACE_VARIANTS,
    gap_opening_cost: float = 1.0,
    gap_extension_cost: float = 0.5,
    gap_power: float = 0.7,
) -> None:
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    trace_variants = list(trace_variants)
    random.shuffle(trace_variants)
    if len(trace_variants) > max_trace_variants:
        trace_variants = trace_variants[OFFSET:(max_trace_variants + OFFSET)]

    print("Align event log using Gap-Aware Alignments")
    reachability_graph = ReachabilityGraph(accepting_petri_net)
    args_iter = (
        (variant, freq, (variant, reachability_graph, gap_opening_cost, gap_extension_cost, gap_power))
        for variant, freq in trace_variants
    )
    with Pool(processes=N_WORKERS) as pool:
        _run_block(
            pool,
            evaluate_trace_gap_aware,
            args_iter,
            result_path / f"result{file_tag}_gap_aware_align.csv",
            repeat,
        )
