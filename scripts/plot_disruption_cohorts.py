"""Plot separate guided/control curves from the completed shutdown sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.online_feedback_experiment import bootstrap


def plot_curves(curves, severities, shutdown_day, output, filename, columns):
    rows = (len(severities) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.7 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, percentage in zip(axes.flat, severities):
        for cohort, label, color in [("pm", "Guided miners", "#0072B2"),
                                      ("control", "Unguided miners (same market)", "#D55E00")]:
            data = curves[(curves.severity_pct == percentage) & (curves.cohort == cohort)].sort_values("day")
            ax.plot(data.day, data.mean_delta, color=color, linewidth=2, label=label)
            ax.fill_between(data.day, data.ci_low, data.ci_high, color=color, alpha=.12)
        ax.axvline(shutdown_day, color="black", linestyle="--", linewidth=1,
                   label=f"Intervention day {shutdown_day}")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_title("0%: no shutdown" if percentage == 0 else f"{percentage:g}% of active miners shut down")
        ax.set_xlim(0, curves.day.max())
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(severities):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3, fontsize=9)
    fig.suptitle("Separate cohort returns under permanent miner shutdowns", fontsize=14)
    fig.supxlabel("Simulated day")
    fig.supylabel("Mean economic return (% of initial net worth)")
    fig.text(.5, .035, "Eight seeds; complementary assignments averaged within seed. Bands: descriptive 95% seed-bootstrap intervals.\n"
             "Each cohort includes all 20 original miners, including exits and forced closures. These are performance curves, not isolated learning effects.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(.02, .085, 1, .92))
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"{filename}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="snapshots.csv from a completed shutdown run")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the regenerated cohort plots")
    parser.add_argument("--shutdown-day", type=int, default=360)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    data = data[(data.family == "process") & data.cohort.isin(["pm", "control"])]
    if data.duplicated(["seed", "severity_pct", "swap", "day", "cohort"]).any():
        raise ValueError("Duplicate cohort checkpoints")
    counts = data.groupby(["seed", "severity_pct", "day", "cohort"]).swap.nunique()
    if not counts.eq(2).all() or not np.isfinite(data.mean_return_pct).all():
        raise ValueError("Need two complementary assignments and finite outcomes at every checkpoint")
    seeds = data.groupby(["seed", "severity_pct", "day", "cohort"], as_index=False).mean_return_pct.mean()
    seeds.to_csv(args.output / "cohort_curve_seed_values.csv", index=False)
    rows = [dict(severity_pct=pct, day=day, cohort=cohort, **bootstrap(group.mean_return_pct))
            for (pct, day, cohort), group in seeds.groupby(["severity_pct", "day", "cohort"])]
    curves = pd.DataFrame(rows)
    curves.to_csv(args.output / "cohort_return_curves.csv", index=False)
    final = seeds[seeds.day == seeds.day.max()].pivot(index=["seed", "severity_pct"],
                                                    columns="cohort", values="mean_return_pct")
    baseline = final.xs(0, level="severity_pct")
    changes = []
    for (seed, pct), row in final.iterrows():
        guided_change = row.pm - baseline.loc[seed, "pm"]
        control_change = row.control - baseline.loc[seed, "control"]
        changes.append(dict(seed=seed, severity_pct=pct,
            guided_return=row.pm, unguided_return=row.control,
            guided_change=guided_change, unguided_change=control_change,
            advantage=row.pm - row.control, advantage_change=guided_change - control_change))
    changes = pd.DataFrame(changes)
    changes.to_csv(args.output / "cohort_change_seed_values.csv", index=False)
    change_effects = pd.DataFrame([
        dict(severity_pct=pct, metric=metric, **bootstrap(group[metric]))
        for pct, group in changes.groupby("severity_pct")
        for metric in ["guided_return", "unguided_return", "guided_change", "unguided_change", "advantage", "advantage_change"]])
    change_effects.to_csv(args.output / "cohort_final_comparison.csv", index=False)
    percentages = sorted(curves.severity_pct.unique())
    focus = [p for p in [0, 5, 10, 100] if p in percentages]
    plot_curves(curves, focus, args.shutdown_day, args.output, "cohort_returns_focus", 2)
    plot_curves(curves, percentages, args.shutdown_day, args.output, "cohort_returns_all", 3)
    provenance = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                      plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      independent_seeds=int(seeds.seed.nunique()),
                      aggregation="Average complementary assignments within seed, then bootstrap seeds",
                      simulation_rerun=False)
    (args.output / "cohort_plot_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(change_effects.pivot(index="severity_pct", columns="metric", values="mean_delta").round(3).to_string())
    print(f"Saved separate cohort curves and comparisons to {args.output}")


if __name__ == "__main__":
    main()
