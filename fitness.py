from __future__ import annotations

import ast
import csv
import math
import sys
from pathlib import Path

from gap_aware_alignments import ReachabilityGraph, import_petri_net

DATA_PATH = Path("data")

C_OPENING = 1.0
C_EXTENSION = 0.5
D_POWER = 0.7

MODELS = ["Linear", "Affine", "Logarithmic", "Power"]


def cost_model(n: float, model: str) -> float:
    """Reference (worst-case) cost g(n) for a single gap of length n."""
    if model == "Linear":
        return n
    if model == "Affine":
        return C_OPENING + C_EXTENSION * n
    if model == "Logarithmic":
        return C_OPENING + math.log(n + 1) * C_EXTENSION
    if model == "Power":
        return C_OPENING + C_EXTENSION * (n ** D_POWER)
    raise ValueError(f"Unknown model: {model}")


def best_worst_cost_for(pnml_path: Path) -> int:
    net = import_petri_net(str(pnml_path))
    return ReachabilityGraph(net).best_worst_cost


def process_result_csv(csv_path: Path) -> None:
    stem = csv_path.name[len("result"):-len("_gap_aware_align.csv")]  # e.g. "_pn30" or ""
    log_name = csv_path.parent.name
    pnml_path = DATA_PATH / f"{log_name}{stem}.pnml"
    if not pnml_path.is_file():
        print(f"  ! skipping {csv_path.name}: no matching Petri net {pnml_path}")
        return

    best_worst_cost = best_worst_cost_for(pnml_path)
    out_path = csv_path.with_name(f"result{stem}_fitness.csv")

    weighted_cost = {m: 0.0 for m in MODELS}
    weighted_fitness = {m: 0.0 for m in MODELS}
    total_freq = 0

    with open(csv_path, newline="") as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.reader(f_in, delimiter="\t")
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(
            ["frequency", "variant_length", "variant"]
            + [f"cost_{m}" for m in MODELS]
            + [f"fitness_{m}" for m in MODELS]
        )
        for row in reader:
            cost_field, _runtime, freq, length, variant = row[0], row[1], row[2], row[3], row[4]
            if cost_field == "timeout":
                continue
            costs = ast.literal_eval(cost_field)  # (linear, affine, log, power)
            freq = int(freq)
            length = int(length)
            reference_length = length + best_worst_cost

            fitness_vals = []
            for i, model in enumerate(MODELS):
                reference = cost_model(reference_length, model)
                fitness = 1 - (costs[i] / reference) if reference > 0 else 1.0
                fitness = min(1.0, max(0.0, fitness))
                fitness_vals.append(fitness)
                weighted_cost[model] += costs[i] * freq
                weighted_fitness[model] += fitness * freq
            total_freq += freq

            writer.writerow(
                [freq, length, variant] + [costs[i] for i in range(4)] + fitness_vals
            )

    print(f"  {csv_path.name}: |rho_min|={best_worst_cost}, {total_freq} weighted traces -> {out_path.name}")
    if total_freq:
        header = "    " + "".join(f"{m:>14}" for m in MODELS)
        cost_line = "    cost " + "".join(f"{weighted_cost[m] / total_freq:>14.3f}" for m in MODELS)
        fit_line = "    fit  " + "".join(f"{weighted_fitness[m] / total_freq:>14.3f}" for m in MODELS)
        print(header)
        print(cost_line)
        print(fit_line)


def latest_run() -> Path | None:
    """Most recent output/<timestamp>/ run directory (digit-named)."""
    runs = [d for d in Path("output").glob("*") if d.is_dir() and d.name.isdigit()]
    return max(runs, key=lambda d: d.name) if runs else None


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__)
        return 1
    if len(sys.argv) == 2:
        target = Path(sys.argv[1])
    else:
        # No argument: default to the latest run under output/.
        target = latest_run()
        if target is None:
            print("No output/<timestamp> run found. Run main.py first.")
            return 1
        print(f"Using latest run: {target}")
    csv_files = sorted(target.rglob("result*_gap_aware_align.csv"))
    if not csv_files:
        print(f"No result*_gap_aware_align.csv found under {target}")
        return 1
    current_log = None
    for csv_path in csv_files:
        if csv_path.parent.name != current_log:
            current_log = csv_path.parent.name
            print(f"\n{current_log}")
        process_result_csv(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
