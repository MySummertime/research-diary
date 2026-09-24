"""Plot an existing fleet-size sensitivity CSV without rerunning the solver."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.plot_helpers import dual_y_axis, stacked_cost_bar


def plot_from_csv(csv_path: str, output_dir: str | None = None) -> None:
    output_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("status") not in {"infeasible", "error"}]
    if not rows:
        raise ValueError(f"No plottable rows in {csv_path}")
    # Preserve non-integral values if present, but render ordinary fleet counts
    # without an artificial decimal suffix (10 rather than 10.0).
    x = [
        int(value) if value.is_integer() else value
        for value in (float(r["fleet_size"]) for r in rows)
    ]
    dual_y_axis(
        x_label="Fleet size", x_vals=x,
        r_risk_vals=[float(r["network_risk_reliability"]) for r in rows],
        r_time_vals=[float(r["network_time_reliability"]) for r in rows],
        save_path=os.path.join(output_dir, "fleet_sensitivity"),
        y1_label=r"$R_{risk}$", y2_label=r"$R_{time}$",
        y1_title="Risk reliability", y2_title="Time reliability")
    dual_y_axis(
        x_label="Fleet size", x_vals=x,
        r_risk_vals=[float(r["Network_Risk"]) for r in rows],
        r_time_vals=[float(r["Network_Time"]) for r in rows],
        save_path=os.path.join(output_dir, "fleet_sensitivity_raw"),
        y1_label="Risk", y2_label="Time (h)",
        y1_title="Risk", y2_title="Time (h)")
    stacked_cost_bar(
        x_label="Fleet size", x_vals=x,
        cost_vals=[float(r["cost"]) for r in rows],
        transport_vals=[float(r["transport_cost"]) for r in rows],
        transfer_vals=[float(r["transfer_cost"]) for r in rows],
        reposition_vals=[float(r["reposition_cost"]) for r in rows],
        unmet_penalty_vals=[float(r["unmet_penalty"]) for r in rows],
        save_path=os.path.join(output_dir, "fleet_sensitivity_cost"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the fleet experiment.csv")
    parser.add_argument("--output", help="Output directory; defaults to the CSV directory")
    args = parser.parse_args()
    plot_from_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
