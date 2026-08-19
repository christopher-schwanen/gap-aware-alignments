from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from gap_aware_alignments import get_trace_variants, import_petri_net, read_event_log
from evaluation import evaluate_event_log

PN_TAGS = ["_pn30", "_pn50", "_pn70"]
REPEAT = 5


def main() -> int:
    # Data path with input data (xes files)
    data_path = Path("data").resolve()
    # Output path to store the results
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir(parents=True)

    for xes_file in sorted(data_path.glob("*.xes")):
        if not xes_file.is_file():
            continue
        cur_path = result_path / xes_file.stem
        cur_path.mkdir()
        print(f"{xes_file.stem}")
        print(" -> importing event log with Rust4PM ...")
        event_log = read_event_log(xes_file)
        trace_variants = get_trace_variants(event_log)
        print(f" -> {len(trace_variants)} trace variants")

        # Determine which Petri net(s) to evaluate against.
        benchmarks: list[tuple[str, Path]] = []
        if (pnml_file := data_path / f"{xes_file.stem}.pnml").is_file():
            benchmarks.append(("", pnml_file))
        else:
            for tag in PN_TAGS:
                pnml_file = data_path / f"{xes_file.stem}{tag}.pnml"
                if not pnml_file.is_file():
                    raise FileNotFoundError(f"Expected Petri net not found: {pnml_file}")
                benchmarks.append((tag, pnml_file))

        for file_tag, pnml_file in benchmarks:
            print(f" -> {pnml_file.stem} (Rust4PM PNML import)")
            accepting_petri_net = import_petri_net(pnml_file)
            evaluate_event_log(
                trace_variants=trace_variants,
                accepting_petri_net=accepting_petri_net,
                repeat=REPEAT,
                result_path=cur_path,
                file_tag=file_tag,
            )

    print(f"\nDone. Results in {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
