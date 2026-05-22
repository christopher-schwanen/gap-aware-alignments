import sys
from datetime import datetime
from pathlib import Path

import pm4py

from evaluation import evaluate_event_log


if __name__ == "__main__":
    # Data path with input data (xes files)
    data_path = Path("data").resolve()
    # Output path to store the results
    result_path = Path("output") / datetime.now().strftime("%Y%m%d%H%M%S")
    result_path.mkdir()

    for xes_file in data_path.glob("*.xes"):
        if not xes_file.is_file():
            continue
        cur_path = result_path / xes_file.stem
        cur_path.mkdir()
        print(f"{xes_file.stem}")
        event_log = pm4py.read_xes(str(xes_file))
        # Compute a list of benchmarks that we should execute
        evaluate_event_logs = []
        # Check if in data_path there is a file with the same name as the xes file but with the extension .ptml
        if (pnml_file := data_path / f"{xes_file.stem}.pnml").is_file():
            accepting_petri_net = pm4py.read_pnml(str(pnml_file))
            print(f" -> {pnml_file.stem}")
            evaluate_event_logs.append({'event_log': event_log,
                                        'accepting_petri_net': accepting_petri_net,
                                        'repeat': 5,
                                        'result_path': cur_path,
                                        'file_tag': ""})
        else:
            for file_tag in ["_pn30", "_pn50", "_pn70"]:
                if (pnml_file := data_path / f"{xes_file.stem}{file_tag}.pnml").is_file():
                    accepting_petri_net = pm4py.read_pnml(str(pnml_file))
                else:
                    raise NotImplementedError()
                    accepting_petri_net = ...
                    pm4py.write_pnml(accepting_petri_net, str(data_path / f"{xes_file.stem}{file_tag}.pnml"))
                print(f" -> {pnml_file.stem}")
                evaluate_event_logs.append({'event_log': event_log,
                                            'accepting_petri_net': accepting_petri_net,
                                            'repeat': 5,
                                            'result_path': cur_path,
                                            'file_tag': file_tag})

        # Evaluation:
        for benchmark in evaluate_event_logs:
            evaluate_event_log(**benchmark, include_pm4py_align=False)

    sys.exit(0)
