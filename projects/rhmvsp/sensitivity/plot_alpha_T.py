"""Plot an existing alpha_T sensitivity CSV without rerunning the solver."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.plot_helpers import dual_y_axis, stacked_cost_bar


def _value(row: dict, *names: str) -> float:
    for name in names:
        if row.get(name) not in {"", None}:
            return float(row[name].replace(",", ""))
    raise KeyError(f"Missing any of columns: {', '.join(names)}")


def plot_from_csv(csv_path: str, output_dir: str | None = None) -> None:
    output_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") not in {"infeasible", "error"}]
    if not rows:
        raise ValueError(f"No plottable rows in {csv_path}")
    x = [_value(r, "alpha_T") for r in rows]
    dual_y_axis(x_label=r"$\alpha_T$", x_vals=x,
                r_risk_vals=[_value(r, "network_risk_reliability", "R_risk") for r in rows],
                r_time_vals=[_value(r, "network_time_reliability", "R_time") for r in rows],
                save_path=os.path.join(output_dir, "sensitivity_alpha_T"),
                y1_label="Risk reliability", y2_label="Time reliability",
                y1_title="Risk reliability", y2_title="Time reliability")
    dual_y_axis(x_label=r"$\alpha_T$", x_vals=x,
                r_risk_vals=[_value(r, "Network_Risk", "Raw_Risk") for r in rows],
                r_time_vals=[_value(r, "Network_Time", "Raw_Time") for r in rows],
                save_path=os.path.join(output_dir, "sensitivity_alpha_T_raw"),
                y1_label="Risk", y2_label="Time (h)",
                y1_title="Risk", y2_title="Time (h)")
    stacked_cost_bar(
        x_label=r"$\alpha_T$", x_vals=x,
        cost_vals=[_value(r, "cost") for r in rows],
        transport_vals=[_value(r, "transport_cost") for r in rows],
        transfer_vals=[_value(r, "transfer_cost") for r in rows],
        reposition_vals=[_value(r, "reposition_cost") for r in rows],
        unmet_penalty_vals=[_value(r, "unmet_penalty") for r in rows],
        save_path=os.path.join(output_dir, "sensitivity_alpha_T_cost"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to summary_alpha_T.csv")
    parser.add_argument("--output", help="Output directory; defaults to the CSV directory")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
