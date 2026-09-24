"""Plot an existing out-of-sample MR/MW experiment CSV."""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.plot_helpers import COLOURS, enhanced_boxplot


METRICS = [
    (("cost",), "Cost", "oos_total_cost", False),
    (("Network_Risk", "network_risk", "Raw_Risk"), "Risk", "oos_network_risk", True),
    (("Network_Time", "network_time", "Raw_Time"), "Time (h)", "oos_raw_Time", True),
    (("network_risk_reliability", "R_risk"), r"Risk Reliability ($R_{risk}$)", "oos_R_Risk", True),
    (("network_time_reliability", "R_time"), r"Time Reliability ($R_{time}$)", "oos_R_Time", True),
]


def plot_from_csv(csv_path: str, output_dir: str | None = None) -> None:
    output_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = list(dict.fromkeys(r["group_label"] for r in rows))
    if not labels:
        raise ValueError(f"No rows in {csv_path}")

    for fields, y_label, filename, impute_no_route in METRICS:
        # Store by scenario so raw-metric imputation preserves the MR/BD
        # pairing used by the Wilcoxon annotation.
        grouped = {"MR": defaultdict(dict), "MW": defaultdict(dict)}
        no_route_scenarios = {"MR": defaultdict(list), "MW": defaultdict(list)}
        for row in rows:
            model = row.get("model")
            label = row.get("group_label")
            if model not in grouped or label not in labels:
                continue
            scenario_id = row.get("scenario_id")
            # The solver writes 0 as a placeholder when it dispatches no
            # route.  Retain that sample, but place it at the median of the
            # corresponding model/parameter group rather than plotting a
            # fictitious zero risk or time.
            if impute_no_route and not row.get("Routes", "").strip():
                no_route_scenarios[model][label].append(scenario_id)
                continue
            value = next((row.get(field) for field in fields if row.get(field) not in {"", None}), "")
            if value not in {"", None}:
                grouped[model][label][scenario_id] = float(value.replace(",", ""))

        if impute_no_route:
            for model, values_by_group in grouped.items():
                for label in labels:
                    observed = values_by_group[label]
                    if no_route_scenarios[model][label] and observed:
                        median = float(np.median(list(observed.values())))
                        for scenario_id in no_route_scenarios[model][label]:
                            observed[scenario_id] = median

        def ordered_values(values_by_scenario: dict[str, float]) -> np.ndarray:
            return np.asarray(
                [values_by_scenario[key] for key in sorted(values_by_scenario, key=int)],
                dtype=float,
            )

        mr = [ordered_values(grouped["MR"][label]) for label in labels]
        mw = [ordered_values(grouped["MW"][label]) for label in labels]
        enhanced_boxplot(
            group_labels=labels, bap_costs=mr, bench_costs=mw,
            save_path=os.path.join(output_dir, filename), y_label=y_label,
            x_label=r"$(\alpha_T,\;\alpha_R)$", model_bap_label="AB-B&P",
            model_bench_label="BD", c_bap=COLOURS["C1"], c_bench=COLOURS["C2"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the out-of-sample experiment.csv")
    parser.add_argument("--output", help="Output directory; defaults to the CSV directory")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
