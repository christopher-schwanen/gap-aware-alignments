# Gap-Aware Alignments

A tool for gap-aware trace-to-model conformance checking, implementing the
methodology from *Improved Process Deviation Diagnostics through Gap-Aware
Alignments* (Morr, Schwanen, van der Aalst). Petri nets (PNML) are the reference
model and event logs (XES) the observed behavior; results are written as CSV.

Instead of penalizing every deviation independently (the classic linear cost),
gap-aware alignments treat a maximal contiguous block of non-synchronous moves
as one *gap* and charge it as a whole. For every trace variant the optimal
alignment cost is computed under four gap cost models in a single pass:

| Model         | Gap cost `g(k)`                       | Reference |
|---------------|---------------------------------------|-----------|
| `classic`     | `k` (linear baseline)                 | Sec. 3.3  |
| `affine`      | `C_open + C_extend · k`                | Def. 6    |
| `logarithmic` | `C_open + C_extend · log(k + 1)`       | Def. 7    |
| `power`       | `C_open + C_extend · k^d`              | Def. 8    |

Default parameters: `C_open = 1.0`, `C_extend = 0.5`, `d = 0.7`.

## How it works

The tool is built on **[Rust4PM](https://rust4pm.aarkue.eu/)** (`r4pm`) for fast
event-data and model I/O, and `networkx` for the shortest-path search:

* **PNML import** → `r4pm.petri_net.import_pnml` (a JSON-compatible dict).
* **XES import** → `r4pm.df.import_xes` (a Polars DataFrame), then grouped into
  trace variants.
* Petri-net **semantics** (enabledness, firing, workflow-net check, artificial
  end transition), the **reachability graph** of the model, and the **gap-aware
  alignment state graph** live in `gap_aware_alignments/`. The state graph pairs
  each trace position with a model marking and materializes the full open-gap
  length on gap edges, so an optimal gap-aware alignment is a shortest path
  (Dijkstra) over the respective cost model.

## Layout

```
gap_aware_alignments/
  petri_net.py            Rust4PM Petri net wrapper + marking/firing semantics
  reachability_graph.py   marking-graph of the accepting Petri net
  gap_aware_alignment.py  alignment state graph + the four cost models
  log_io.py               XES import + trace-variant extraction (r4pm + polars)
evaluation.py             per-variant evaluation, parallel, timeout-guarded -> CSV
main.py                   entry point: iterate data/*.xes and matching PNMLs
fitness.py                turn alignment CSVs into per-variant fitness CSVs (Sec. 3.4)
pyproject.toml / uv.lock  pinned dependency set
data/                     <log>.xes and <log>_pn{30,50,70}.pnml
```

## Input naming convention

Each `data/<log>.xes` is paired with either a single `data/<log>.pnml` or, per
Inductive-Miner-infrequent noise threshold, one net each:
`data/<log>_pn30.pnml`, `data/<log>_pn50.pnml`, `data/<log>_pn70.pnml`
(thresholds 0.3 / 0.5 / 0.7).

## Setup

Requires Python 3.10–3.12 (r4pm ships wheels there; 3.13/3.14 not yet).

With [uv](https://docs.astral.sh/uv/) and the pinned `uv.lock`:

```bash
uv sync
```

Or with a plain virtual environment:

```bash
python3.12 -m venv .venv
./.venv/bin/pip install r4pm polars networkx
```

## Usage

```bash
# Align every log against its Petri net(s).
# Writes output/<timestamp>/<log>/result<tag>_gap_aware_align.csv
python main.py

# Compute normalized fitness (Sec. 3.4) for the latest run.
# Writes result<tag>_fitness.csv per log and prints the weighted mean cost/fitness.
python fitness.py                 # latest run under output/  (or pass output/<timestamp>)
```

## Output CSV

`result<tag>_gap_aware_align.csv` (tab-separated), one row per trace variant:

```
(classic, affine, log, power)  runtime  frequency  length  variant  \
    classic_alignment  affine_alignment  log_alignment  power_alignment
```

* **column 1** — the four optimal alignment costs as a tuple;
* **column 2** — alignment runtime in seconds (fastest of the repeats);
* **column 3** — how many cases in the log follow this variant (frequency);
* **column 4** — number of events in the variant (`|σ|`);
* **column 5** — the variant itself;
* **columns 6–9** — the optimal alignment under each cost model.

`result<tag>_fitness.csv` adds per-model `cost_*` and `fitness_*` columns, where
`fitness_g(σ, M) = 1 − C_g(γ*) / g(|σ| + |ρ_min|) ∈ [0, 1]` and `|ρ_min|` is the
length of a shortest accepted model trace.

## Example results

Frequency-weighted mean cost and fitness on the Road Traffic Fine Management
Process log, model discovered at noise threshold 0.3:

| Model       | Mean cost | Mean fitness |
|-------------|-----------|--------------|
| Linear      | 0.177     | 0.947        |
| Affine      | 0.258     | 0.903        |
| Logarithmic | 0.230     | 0.867        |
| Power       | 0.257     | 0.881        |
