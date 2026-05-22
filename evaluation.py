import csv
import ctypes
import random
import threading
import timeit
from multiprocessing import Pool, cpu_count, TimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Union

from pm4py.algo.conformance.alignments.petri_net.algorithm import (__close_progress_bar as close_progress_bar,
                                                                   __get_progress_bar as get_progress_bar,
                                                                   __get_variants_structure as get_variants,
                                                                   apply as pm4py_align_petri_net)
from pm4py.objects.log.obj import Trace, EventLog

from gap_aware_alignments import ReachabilityGraph, align

if TYPE_CHECKING:
    import pandas as pd

    from pm4py import PetriNet, Marking


# Set seed for reproducibility
random.seed(42)
# Number of CPUs to use
N_CPUS = cpu_count()
N_WORKERS = min(N_CPUS - 2, 61)
TIMEOUT = 65
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


def _timed_call(fn, timeout: float) -> tuple[float, float]:
    """
    Run fn() in a daemon thread and measure wall time with timeit.
    If fn() does not complete within timeout seconds, _AlignmentTimeout is injected
    into the thread and re-raised here. The worker process is not affected.
    Returns (result, elapsed).
    """
    outcome = []  # filled as [cost, elapsed] by the thread
    error = [None]

    def _target():
        try:
            elapsed = timeit.timeit(lambda: outcome.append(fn()), number=1)
            outcome.append(elapsed)  # outcome == [cost, elapsed]
        except _AlignmentTimeout:
            pass
        except Exception as e:
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
        # Thread was interrupted before timeit completed
        raise _AlignmentTimeout()

    return outcome[0], outcome[1]


def evaluate_trace_gap_aware(trace: Trace,
                             reachability_graph: ReachabilityGraph,
                             repeat: int = 10,
                             ) -> list[tuple[float, float]]:
    runs = []
    for _ in range(repeat):
        try:
            cost, elapsed = _timed_call(lambda: align(trace, reachability_graph), TIMEOUT)
            runs.append((cost, elapsed))
        except _AlignmentTimeout:
            break
    return runs


def evaluate_trace_pm4py(trace: Trace,
                         accepting_petri_net: tuple['PetriNet', 'Marking', 'Marking'],
                         repeat: int = 10,
                         ) -> list[tuple[float, float]]:
    runs = []
    for _ in range(repeat):
        try:
            cost, elapsed = _timed_call(lambda: pm4py_align_petri_net(trace, *accepting_petri_net)["cost"], TIMEOUT)
            runs.append((cost, elapsed))
        except _AlignmentTimeout:
            break
    return runs


def _run_block(pool: Pool,
               fn,
               args_iter,
               output_path: Path,
               repeat: int = 10,
               ) -> None:
    """Submit one task per variant, collect best-of-repeat results, write to CSV.

    args_iter yields (variant, freq, args_tuple) where args_tuple is passed to fn
    together with repeat as positional arguments.
    """
    pending = [(variant, freq, pool.apply_async(fn, args=args + (repeat,)))
               for variant, freq, args in args_iter]
    progress_bar = get_progress_bar(len(pending), None)
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        for variant, freq, async_result in pending:
            try:
                runs = async_result.get(timeout=(repeat + 1) * TIMEOUT)  # additional timeout in case _timed_call itself gets stuck
                if not runs:
                    raise TimeoutError
                costs = [c for c, _ in runs]
                if len(set(costs)) > 1:
                    raise ValueError(f"Cost mismatch across repetitions for variant {variant}: {costs}")
                """Write best-of-repeat result to CSV, or a timeout row if no run completed."""
                writer.writerow([costs[0], min(t for _, t in runs), freq, len(variant), variant])
            except TimeoutError:
                writer.writerow(["timeout", "timeout", freq, len(variant), variant])
            finally:
                f.flush()
                progress_bar.update()
    close_progress_bar(progress_bar)
    return


def evaluate_event_log(event_log: Union[EventLog, 'pd.DataFrame'],
                       accepting_petri_net: tuple['PetriNet', 'Marking', 'Marking'],
                       repeat: int = 5,
                       result_path: str | Path = "output",
                       file_tag: str = "",
                       max_trace_variants: int = MAX_TRACE_VARIANTS,
                       include_pm4py_align: bool = False,
                       ) -> None:
    if isinstance(result_path, str):
        result_path = Path(result_path)
    if not result_path.is_dir():
        result_path.mkdir()

    # Get trace variant results and shuffle them
    trace_variants = get_variants(event_log, None)
    trace_variants: list[tuple[tuple[str, ...], int, Trace]] = [(k, len(v), t)
                                                                for (k, v), t
                                                                in zip(trace_variants[0].items(), trace_variants[1])
                                                                ]
    random.shuffle(trace_variants)
    # Check if number of trace_variants is below max_trace_variants
    if len(trace_variants) > max_trace_variants:
        trace_variants = trace_variants[OFFSET:(max_trace_variants + OFFSET)]

    # Gap-Aware Alignment
    if True:
        print("Align event log using Gap-Aware Alignments")
        reachability_graph = ReachabilityGraph(accepting_petri_net)
        args_iter = ((variant, freq, (trace, reachability_graph))
                     for variant, freq, trace in trace_variants)
        with Pool(processes=N_WORKERS) as pool:
            _run_block(pool, evaluate_trace_gap_aware, args_iter, result_path / f"result{file_tag}_gap_aware_align.csv", repeat)

    # PM4Py Alignments
    if include_pm4py_align:
        print("Align event log using PM4Py Petri Net Alignments")
        args_iter = ((variant, freq, (trace, accepting_petri_net))
                     for variant, freq, trace in trace_variants)
        with Pool(processes=N_WORKERS) as pool:
            _run_block(pool, evaluate_trace_pm4py, args_iter, result_path / f"result{file_tag}_pm4py_pn_align.csv", repeat)

    return
