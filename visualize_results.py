"""
visualize_results.py

Auto-discovers all event logs in the output/ directory, loads the
gap-aware alignment CSVs and (where needed) the corresponding PNML files,
then produces the visualisations enabled by the feature flags below.

Output types (toggle via the flags in the "Feature flags" section):

  EXPORT_TIKZ_BOXPLOTS   – cost + fitness boxplots as standalone pgfplots
                           .tex files (one per log & value). This is the
                           paper deliverable: precomputed weighted box
                           statistics + deduplicated outliers whose opacity
                           encodes frequency, so the figure is tiny and
                           lag-free and its fonts match the LaTeX document.
  EXPORT_PDF_BOXPLOTS    – the same boxplots as matplotlib PDF pages.
  EXPORT_HISTOGRAMS      – gap-length distribution (PDF), kept for the thesis.
  EXPORT_PETRI_HEATMAPS  – petri-net deviation heatmaps (PDF), kept for the
                           thesis.

The PDF outputs are bundled into one <log>_evaluation.pdf per event log;
only the enabled page types are written.
"""

import ast
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from gap_aware_alignments.reachability_graph import ReachabilityGraph


# ---------------------------------------------------------------------------
# Feature flags – decide what this run generates
# ---------------------------------------------------------------------------

EXPORT_TIKZ_BOXPLOTS  = True    # cost + fitness boxplots as pgfplots .tex (paper)
EXPORT_PDF_BOXPLOTS   = False   # cost + fitness boxplots as PDF pages
EXPORT_HISTOGRAMS     = False   # gap-length distribution (PDF, for thesis)
EXPORT_PETRI_HEATMAPS = False   # petri-net deviation heatmaps (PDF, for thesis)
PRINT_STATS           = True    # print weighted box statistics to the console


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLDS      = [0.3, 0.5, 0.7]
THRESHOLD_TAGS  = {0.3: "pn30", 0.5: "pn50", 0.7: "pn70"}

C_OPENING   = 1.0
C_EXTENSION = 0.5
D_POWER     = 0.7

# Base per-marker opacity of a single outlier occurrence. Stacking n of them
# yields accumulated opacity 1-(1-FLIER_ALPHA)^n, which we reproduce with a
# single marker per distinct value (see weighted_boxplot_stats).
FLIER_ALPHA = 0.4

# Layout of the TikZ boxplots. These reproduce the hand-tuned style; adjust
# here and every log comes out consistently.
TIKZ_PANEL_WIDTH  = "5cm"          # width of one threshold panel
TIKZ_PANEL_HEIGHT = "5.5cm"        # height of one threshold panel
TIKZ_OUTLIER_SIZE = "0.7pt"        # marker size of the outlier dots
TIKZ_MEDIAN_WIDTH = "0.6pt"        # line width of the (orange) median
TIKZ_XTICKLABELS  = "Linear,Affine,Log,Power"  # abbreviated for narrow panels
TIKZ_Y_GAP_FRAC   = 0.05           # empty space below 0, as fraction of ymax

COST_MODEL_LABELS = ["Linear", "Affine", "Logarithmic", "Power"]
COST_MODEL_KEYS   = ["Linear", "Affine", "Logarithmic", "Power (d=0.7)"]
ALIGNMENT_COLS    = {
    "Linear":        "alignment_linear",
    "Affine":        "alignment_affine",
    "Logarithmic":   "alignment_log",
    "Power (d=0.7)": "alignment_power",
}

# One distinct colour per cost model
MODEL_COLORS = {
    "Linear":        "#1f77b4",
    "Affine":        "#ff7f0e",
    "Logarithmic":   "#2ca02c",
    "Power (d=0.7)": "#d62728",
}

OUTPUT_DIR   = Path("output")
DATA_DIR     = Path("data")
PDF_OUT_DIR  = Path("/tmp/gap_aware_eval")   # local, avoids iCloud write stalls
TIKZ_OUT_DIR = Path("output/tikz")           # .tex files are tiny, safe in project


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_event_logs() -> dict[str, dict[float, Path]]:
    """
    Scans output/<timestamp>/<EventLogName>/ for result CSVs.
    Returns {log_name: {threshold: csv_path}} using the most recent
    timestamp for each (log_name, threshold) pair.
    """
    # log_name -> threshold -> [(timestamp_str, path)]
    entries: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))

    for ts_dir in OUTPUT_DIR.iterdir():
        if not ts_dir.is_dir() or not ts_dir.name.isdigit():
            continue
        for log_dir in ts_dir.iterdir():
            if not log_dir.is_dir():
                continue
            log_name = log_dir.name
            for threshold, tag in THRESHOLD_TAGS.items():
                csv_path = log_dir / f"result_{tag}_gap_aware_align.csv"
                if csv_path.exists():
                    entries[log_name][threshold].append((ts_dir.name, csv_path))

    # Keep only the latest run per (log, threshold)
    result: dict[str, dict[float, Path]] = {}
    for log_name, threshold_map in entries.items():
        result[log_name] = {}
        for threshold, ts_paths in threshold_map.items():
            latest_path = max(ts_paths, key=lambda x: x[0])[1]
            result[log_name][threshold] = latest_path

    return result


def find_pnml(log_name: str, threshold: float) -> Path | None:
    tag  = THRESHOLD_TAGS[threshold]
    path = DATA_DIR / f"{log_name}_{tag}.pnml"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Cost model helpers (same as calc_fitness.py)
# ---------------------------------------------------------------------------

def cost_model_fn(n: float, model: str) -> float:
    if model == "Linear":
        return n
    if model == "Affine":
        return C_OPENING + C_EXTENSION * n
    if model == "Logarithmic":
        return C_OPENING + math.log1p(n) * C_EXTENSION
    if model == "Power (d=0.7)":
        return C_OPENING + C_EXTENSION * (n ** D_POWER)
    raise ValueError(f"Unknown model: {model}")


def compute_best_worst_cost(pnml_path: Path) -> float:
    net, im, fm = pnml_importer.apply(str(pnml_path))
    return ReachabilityGraph((net, im, fm)).best_worst_cost


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_result_csv(csv_path: Path, best_worst_cost: float) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep="\t",
        header=None,
        names=[
            "cost_tuple", "runtime", "frequency", "variant_length", "variant",
            "alignment_linear", "alignment_affine", "alignment_log", "alignment_power",
        ],
    )

    # Drop timeout rows
    df = df[df["cost_tuple"] != "timeout"].copy()

    df["cost_tuple"] = df["cost_tuple"].apply(ast.literal_eval)
    df["frequency"]  = df["frequency"].astype(int)

    for i, model in enumerate(COST_MODEL_KEYS):
        df[f"cost_{model}"] = df["cost_tuple"].apply(lambda x, i=i: x[i])

    df["reference_length"] = df["variant_length"] + best_worst_cost
    for model in COST_MODEL_KEYS:
        ref = df["reference_length"].apply(lambda n: cost_model_fn(n, model))
        df[f"reference_{model}"] = ref
        df[f"fitness_{model}"]   = (1 - df[f"cost_{model}"] / ref).clip(0, 1)

    # Parse alignment columns
    for col in ALIGNMENT_COLS.values():
        df[col] = df[col].apply(ast.literal_eval)

    return df


def weighted_values(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.repeat(
        df[column].to_numpy(dtype=float),
        df["frequency"].to_numpy(dtype=int),
    )


# ---------------------------------------------------------------------------
# Weighted boxplot statistics (for TikZ export)
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    """Compact number formatting for the generated .tex."""
    return f"{float(x):.6g}"


def weighted_boxplot_stats(values: np.ndarray,
                           weights: np.ndarray,
                           alpha: float = FLIER_ALPHA) -> dict:
    """
    Compute frequency-weighted box statistics plus deduplicated outliers.

    The box (median, quartiles, whiskers, mean) is computed on the
    frequency-expanded sample, so it is identical to what matplotlib's
    boxplot produces on weighted_values() — the numbers match the PDF plots.

    Outliers are returned deduplicated: one entry (value, n, opacity) per
    distinct value, where n is the summed frequency and

        opacity = 1 - (1 - alpha)^n

    is exactly the opacity that n stacked alpha-transparent markers would
    accumulate. Rendering one marker with this opacity reproduces the
    "darker = more frequent" look without drawing thousands of stacked
    points, which is what made the embedded PDF lag.
    """
    values  = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=int)

    expanded = np.repeat(values, weights)
    q1, median, q3 = (float(v) for v in np.quantile(expanded, [0.25, 0.5, 0.75]))
    iqr      = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr

    within     = expanded[(expanded >= lo_fence) & (expanded <= hi_fence)]
    lo_whisker = float(within.min()) if within.size else q1
    hi_whisker = float(within.max()) if within.size else q3
    mean       = float(expanded.mean())

    # Dedup outliers by distinct value; accumulate frequency for opacity.
    grouped: dict[float, int] = defaultdict(int)
    mask = (values < lo_fence) | (values > hi_fence)
    for val, wt in zip(values[mask], weights[mask]):
        grouped[float(val)] += int(wt)

    outliers = []
    for val, n in sorted(grouped.items()):
        opacity = 1.0 - (1.0 - alpha) ** n
        outliers.append((val, n, opacity))

    return {
        "q1": q1, "median": median, "q3": q3,
        "lo_whisker": lo_whisker, "hi_whisker": hi_whisker,
        "mean": mean, "outliers": outliers,
    }


# ---------------------------------------------------------------------------
# TikZ / pgfplots export
# ---------------------------------------------------------------------------

def _tikz_boxplot(data: dict[float, pd.DataFrame],
                  available: list[float],
                  value: str,          # "cost" or "fitness"
                  log_name: str,
                  out_path: Path) -> None:
    """
    Write one standalone pgfplots figure (a groupplot with one panel per
    noise threshold, four boxes per panel) to out_path as a .tex fragment
    meant to be \\input into the paper.
    """
    value_label = "Fitness" if value == "fitness" else "Alignment Cost"
    n_panels    = len(available)

    lines: list[str] = [
        "% Auto-generated by visualize_results.py — do not edit by hand.",
        "% Required in the LaTeX preamble:",
        "%   \\usepackage{pgfplots}",
        "%   \\pgfplotsset{compat=1.18}",
        "%   \\usepgfplotslibrary{groupplots}",
        "%   \\usepgfplotslibrary{statistics}",
        "\\begin{tikzpicture}",
    ]

    # Shared y-range across all panels. groupplot does NOT equalise the y-axis
    # on its own — without this each panel autoscales independently, so with
    # the tick labels hidden the boxes would look comparable while actually
    # sitting on different scales. ymin is a small negative offset so the 0
    # line sits slightly above the axis instead of flush on it.
    if value == "fitness":
        ymax = 1.0
    else:
        global_max = max(
            float(data[t][f"cost_{m}"].max())
            for t in available for m in COST_MODEL_KEYS
        )
        ymax = math.ceil(global_max) + 1
    ymin = -TIKZ_Y_GAP_FRAC * ymax

    axis_opts = [
        f"group style={{group size={n_panels} by 1, ylabels at=edge left, "
        "yticklabels at=edge left, horizontal sep=0.4cm}",
        f"width={TIKZ_PANEL_WIDTH}, height={TIKZ_PANEL_HEIGHT}",
        "boxplot/draw direction=y",
        "boxplot/box extend=0.5",
        f"boxplot/every median/.style={{orange, line width={TIKZ_MEDIAN_WIDTH}}}",
        f"ymin={_fmt(ymin)}, ymax={_fmt(ymax)}",
        "enlarge y limits=false",
        "clip=false",
        "xtick={1,2,3,4}",
        "xmin=0.4, xmax=4.6",
        f"xticklabels={{{TIKZ_XTICKLABELS}}}",
        "x tick label style={rotate=30, anchor=north, font=\\tiny, yshift=-0.1cm}",
        "ylabel style={font=\\scriptsize}",
        "ymajorgrids, yminorgrids, minor y tick num=1",
        "major grid style={gray!25}, minor grid style={gray!12}",
        "tick align=outside",
        "ytick pos=left",
        "xtick pos=bottom",
    ]

    lines.append("\\begin{groupplot}[\n  " + ",\n  ".join(axis_opts) + "\n]")

    for panel_idx, threshold in enumerate(available):
        df   = data[threshold]
        opts = [f"title={{Noise threshold = {threshold}}}"]
        if panel_idx == 0:
            opts.append(f"ylabel={{{value_label}}}")
        lines.append(f"\\nextgroupplot[{', '.join(opts)}]")

        for pos, model in enumerate(COST_MODEL_KEYS, start=1):
            stats = weighted_boxplot_stats(
                df[f"{value}_{model}"].to_numpy(dtype=float),
                df["frequency"].to_numpy(dtype=int),
            )

            # The box itself (precomputed five-number summary).
            lines.append(
                "\\addplot[black, solid, mark=none, boxplot prepared={"
                f"draw position={pos}, "
                f"lower whisker={_fmt(stats['lo_whisker'])}, "
                f"lower quartile={_fmt(stats['q1'])}, "
                f"median={_fmt(stats['median'])}, "
                f"upper quartile={_fmt(stats['q3'])}, "
                f"upper whisker={_fmt(stats['hi_whisker'])}"
                "}] coordinates {};"
            )

            # Mean marker (mirrors the dashed mean line of the PDF version).
            lines.append(
                "\\addplot[only marks, mark=-, mark size=6pt, green!55!black, "
                f"forget plot] coordinates {{({pos},{_fmt(stats['mean'])})}};"
            )

            # Deduplicated outliers, grouped by shared opacity so all the
            # frequency-1 outliers collapse into a single \addplot.
            by_opacity: dict[float, list[float]] = defaultdict(list)
            for val, _n, op in stats["outliers"]:
                by_opacity[round(op, 4)].append(val)
            for op, vals in sorted(by_opacity.items()):
                coords = " ".join(f"({pos},{_fmt(v)})" for v in vals)
                lines.append(
                    f"\\addplot[only marks, mark=*, mark size={TIKZ_OUTLIER_SIZE}, "
                    "mark options={draw=black, fill=black}, "
                    f"opacity={_fmt(op)}, forget plot] coordinates {{{coords}}};"
                )

    lines.append("\\end{groupplot}")
    lines.append("\\end{tikzpicture}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Console statistics table
# ---------------------------------------------------------------------------

def print_boxplot_stats(data: dict[float, pd.DataFrame],
                        available: list[float],
                        log_name: str) -> None:
    """Print the weighted box statistics per threshold / cost model, so the
    exact numbers (mean, median, quartiles, whiskers) can be cited in text."""
    print(f"\n----- statistics: {log_name} -----")
    for threshold in available:
        df         = data[threshold]
        n_variants = len(df)
        n_traces   = int(df["frequency"].sum())
        for value in ("cost", "fitness"):
            print(f"\n  [{value}]  threshold={threshold}  "
                  f"variants={n_variants}  weighted_n={n_traces}")
            print(f"    {'model':<14}{'mean':>8}{'median':>8}{'Q1':>8}"
                  f"{'Q3':>8}{'loWhis':>8}{'hiWhis':>8}{'#outl':>7}")
            for model in COST_MODEL_KEYS:
                s = weighted_boxplot_stats(
                    df[f"{value}_{model}"].to_numpy(dtype=float),
                    df["frequency"].to_numpy(dtype=int),
                )
                n_outl = sum(n for _, n, _ in s["outliers"])
                print(f"    {model:<14}{s['mean']:>8.3f}{s['median']:>8.3f}"
                      f"{s['q1']:>8.3f}{s['q3']:>8.3f}{s['lo_whisker']:>8.3f}"
                      f"{s['hi_whisker']:>8.3f}{n_outl:>7}")


# ---------------------------------------------------------------------------
# Gap-length extraction
# ---------------------------------------------------------------------------

def get_gap_lengths(alignment_moves: list) -> list[int]:
    """
    Returns a list of gap lengths.
    A gap is a maximal consecutive run of non-sync moves
    (i.e. moves where log == '>>' or model == '>>').
    """
    gap_lengths = []
    current     = 0
    for log_label, model_label in alignment_moves:
        if log_label == ">>" or model_label == ">>":
            current += 1
        else:
            if current > 0:
                gap_lengths.append(current)
                current = 0
    if current > 0:
        gap_lengths.append(current)
    return gap_lengths


def extract_gap_lengths_weighted(df: pd.DataFrame, alignment_col: str) -> list[int]:
    """Frequency-weighted list of all gap lengths across all traces."""
    all_gaps = []
    for _, row in df.iterrows():
        gaps = get_gap_lengths(row[alignment_col])
        all_gaps.extend(gaps * int(row["frequency"]))
    return all_gaps


# ---------------------------------------------------------------------------
# Deviation counts per transition
# ---------------------------------------------------------------------------

def count_model_deviations(df: pd.DataFrame, alignment_col: str) -> dict[str, float]:
    """
    For each model-move ('>>', transition_label) compute the deviation rate
    across variants (not weighted by frequency):

        deviation_rate = number of variants with a model-move at transition X
                         ──────────────────────────────────────────────────────
                         total number of deviating variants (cost > 0)

    Each variant counts once regardless of how often it occurs in the log,
    so high-frequency variants don't dominate the heatmap.
    """
    cost_col = "cost_" + {
        "alignment_linear":  "Linear",
        "alignment_affine":  "Affine",
        "alignment_log":     "Logarithmic",
        "alignment_power":   "Power (d=0.7)",
    }[alignment_col]

    df_dev = df[df[cost_col] > 0]
    total_deviating_variants = len(df_dev)

    if total_deviating_variants == 0:
        return {}

    counts: dict[str, int] = defaultdict(int)
    for _, row in df_dev.iterrows():
        for log_label, model_label in row[alignment_col]:
            if log_label == ">>" and model_label != ">>":
                counts[model_label] += 1  # jede Variant zählt einmal

    return {t: c / total_deviating_variants for t, c in counts.items()}


# ---------------------------------------------------------------------------
# Petri-net heatmap rendering
# ---------------------------------------------------------------------------

def _deviation_color(count: int, max_count: int) -> str:
    """Green (0 deviations) → yellow → dark red (max deviations)."""
    if max_count == 0:
        return "#2ca02c"
    ratio = min(count / max_count, 1.0)
    cmap  = mcolors.LinearSegmentedColormap.from_list(
        "dev", ["#2ca02c", "#ffdd57", "#8B0000"]
    )
    r, g, b, _ = cmap(ratio)
    return mcolors.to_hex((r, g, b))


def render_pn_heatmap_to_ax(
    net, im, fm,
    deviation_counts: dict[str, int],
    ax: plt.Axes,
    title: str,
) -> None:
    """
    Renders a colour-coded Petri net via PM4Py/Graphviz to a temp PNG
    and places the image on the given matplotlib Axes.
    """
    max_count = max(deviation_counts.values(), default=0)

    # Build PM4Py decorations: colour each transition by deviation count
    decorations = {}
    for t in net.transitions:
        label  = t.label if t.label else t.name
        count  = deviation_counts.get(label, 0)
        color  = _deviation_color(count, max_count)
        decorations[t] = {"color": color, "label": label or "τ"}

    gviz = pn_visualizer.apply(
        net, im, fm,
        parameters={"format": "png", "decorations": decorations},
    )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pn_visualizer.save(gviz, tmp_path)
        img = plt.imread(tmp_path)
        ax.imshow(img)
    finally:
        os.unlink(tmp_path)

    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)


def _colorbar_legend(ax: plt.Axes, max_count: int) -> None:
    """Adds a small gradient legend (green → dark red) to the axes."""
    cmap   = mcolors.LinearSegmentedColormap.from_list(
        "dev", ["#2ca02c", "#ffdd57", "#8B0000"]
    )
    sm     = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count or 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02,
                 label="Deviation rate (per trace)")


# ---------------------------------------------------------------------------
# PDF plotting helpers
# ---------------------------------------------------------------------------

def _boxplot_page(
    pdf: PdfPages,
    data: dict[float, pd.DataFrame],
    available: list[float],
    value: str,         # "cost" or "fitness"
    log_name: str,
) -> None:
    n      = len(available)
    ylabel = "Fitness" if value == "fitness" else "Alignment Cost"
    title  = f"{log_name} — {ylabel}"

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)

    for ax, threshold in zip(axes, available):
        df        = data[threshold]
        box_data  = [weighted_values(df, f"{value}_{m}") for m in COST_MODEL_KEYS]
        ax.boxplot(
            box_data,
            tick_labels=COST_MODEL_LABELS,
            showfliers=True,
            showmeans=True,
            meanline=True,
            flierprops=dict(marker="o", markersize=2, alpha=FLIER_ALPHA),
        )
        ax.set_title(f"Noise threshold = {threshold}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
        if value == "fitness":
            ax.set_ylim(0, 1)

    axes[0].set_ylabel(ylabel)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _histogram_page(
    pdf: PdfPages,
    data: dict[float, pd.DataFrame],
    available: list[float],
    log_name: str,
) -> None:
    n   = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{log_name} — Gap Length Distribution", fontsize=14)

    for ax, threshold in zip(axes, available):
        df      = data[threshold]
        max_gap = 0

        # First pass: find the global max gap for this threshold
        for col in ALIGNMENT_COLS.values():
            gaps = extract_gap_lengths_weighted(df, col)
            if gaps:
                max_gap = max(max_gap, max(gaps))

        bins = list(range(1, max_gap + 2)) if max_gap > 0 else [1, 2]

        for model, col in ALIGNMENT_COLS.items():
            gaps  = extract_gap_lengths_weighted(df, col)
            color = MODEL_COLORS[model]
            if gaps:
                ax.hist(
                    gaps,
                    bins=bins,
                    alpha=0.55,
                    color=color,
                    label=model,
                    align="left",
                    rwidth=0.8,
                )

        ax.set_title(f"Noise threshold = {threshold}")
        ax.set_xlabel("Gap Length")
        ax.set_ylabel("Weighted Count (log scale)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _petri_net_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    net, im, fm,
    threshold: float,
    log_name: str,
) -> None:
    """One landscape page with a 2×2 grid of Petri net heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{log_name} — Petri Net Deviation Heatmap  (noise threshold = {threshold})",
        fontsize=13,
    )

    items = list(ALIGNMENT_COLS.items())
    for idx, (model, col) in enumerate(items):
        ax             = axes[idx // 2][idx % 2]
        dev_counts     = count_model_deviations(df, col)
        render_pn_heatmap_to_ax(net, im, fm, dev_counts, ax, model)
        _colorbar_legend(ax, max(dev_counts.values(), default=0))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, orientation="landscape")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-log pipeline
# ---------------------------------------------------------------------------

def visualize_event_log(
    log_name: str,
    csv_files: dict[float, Path],
    pnml_files: dict[float, Path],
) -> None:
    print(f"\n[{log_name}]")

    # ------------------------------------------------------------------
    # Load data. The Petri net objects are only needed for the heatmaps,
    # so we only import them when that output is enabled.
    # ------------------------------------------------------------------
    need_nets = EXPORT_PETRI_HEATMAPS
    data: dict[float, pd.DataFrame] = {}
    nets: dict[float, tuple]        = {}

    for threshold in THRESHOLDS:
        if threshold not in csv_files:
            print(f"  ⚠  CSV missing for threshold {threshold}, skipping.")
            continue
        if threshold not in pnml_files:
            print(f"  ⚠  PNML missing for threshold {threshold}, skipping.")
            continue

        print(f"  Loading threshold {threshold} …", end=" ", flush=True)
        bwc              = compute_best_worst_cost(pnml_files[threshold])
        data[threshold]  = load_result_csv(csv_files[threshold], bwc)
        if need_nets:
            nets[threshold] = pnml_importer.apply(str(pnml_files[threshold]))
        print(f"{len(data[threshold])} variants loaded  (bwc={bwc:.3f})")

    available = sorted(data.keys())
    if not available:
        print("  No data — skipping.")
        return

    # ------------------------------------------------------------------
    # Console statistics
    # ------------------------------------------------------------------
    if PRINT_STATS:
        print_boxplot_stats(data, available, log_name)

    # ------------------------------------------------------------------
    # TikZ boxplots (paper deliverable)
    # ------------------------------------------------------------------
    if EXPORT_TIKZ_BOXPLOTS:
        for value in ("cost", "fitness"):
            out = TIKZ_OUT_DIR / f"{log_name}_{value}_boxplot.tex"
            _tikz_boxplot(data, available, value, log_name, out)
            print(f"  ✓  TikZ → {out}")

    # ------------------------------------------------------------------
    # PDF outputs (thesis). Only build the PDF if at least one PDF page
    # type is enabled, and only write the enabled page types.
    # ------------------------------------------------------------------
    if EXPORT_PDF_BOXPLOTS or EXPORT_HISTOGRAMS or EXPORT_PETRI_HEATMAPS:
        output_pdf = PDF_OUT_DIR / f"{log_name}_evaluation.pdf"
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(output_pdf) as pdf:
            if EXPORT_PDF_BOXPLOTS:
                _boxplot_page(pdf, data, available, "cost", log_name)
                _boxplot_page(pdf, data, available, "fitness", log_name)

            if EXPORT_HISTOGRAMS:
                _histogram_page(pdf, data, available, log_name)

            if EXPORT_PETRI_HEATMAPS:
                for threshold in available:
                    net, im, fm = nets[threshold]
                    _petri_net_page(pdf, data[threshold], net, im, fm, threshold, log_name)

        print(f"  ✓  PDF → {output_pdf}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    event_logs = discover_event_logs()

    if not event_logs:
        print("No event logs found in output/. Nothing to do.")
    else:
        print(f"Found {len(event_logs)} event log(s): {', '.join(event_logs)}")

        for log_name, csv_map in event_logs.items():
            pnml_map = {
                t: p
                for t in csv_map
                if (p := find_pnml(log_name, t)) is not None
            }
            visualize_event_log(log_name, csv_map, pnml_map)

    print("\nDone.")
