"""Reproduce the paper's descriptive baseline and Figure 2 without Jupyter."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from scripts.run_feedback_experiment import ROOT, load_model
from scripts.notebook_exports import incoming_edge_counts
from pm4py.algo.filtering.dfg import dfg_filtering
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output_mas/current/baseline"))
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory; existing results are preserved")
    output.mkdir(parents=True, exist_ok=True)
    _, scope = load_model()
    model = scope.AdaptiveMiningModel(scope.params, record_agent_history=True)
    model.run_model()
    agents = model.agent_summary_df().set_index("agent_id")
    events = model.event_log_df().sort_values(["agent_id", "cycle", "seq"])
    snapshots = pd.DataFrame(model.agent_history)
    capital = agents.capital
    cohorts = {
        "top": capital[capital >= capital.quantile(.9)].index,
        "mid": capital[capital.between(capital.quantile(.25), capital.quantile(.75))].index,
        "bot": capital[capital <= capital.quantile(.1)].index,
    }
    trajectories = {}
    # Figure 2 uses each agent's last recorded event within a day.
    event_states = events.groupby(["cycle", "agent_id"]).last().reset_index()
    for name, ids in cohorts.items():
        grouped = event_states[event_states.agent_id.isin(ids)].groupby("cycle").capital.agg(["mean", "std"])
        trajectories[name] = grouped.rename(columns=lambda col: f"{name}_{col}")
    trajectory = pd.concat(trajectories.values(), axis=1)
    counts = pd.DataFrame({
        label: events[events.agent_id.isin(cohorts[key])].activity.value_counts()
        for label, key in [("top_10pct", "top"), ("bottom_10pct", "bot")]
    }).fillna(0).astype(int).rename_axis("activity")
    summary = {
        "agents": len(agents), "days": scope.params.cycles,
        "survivors": int((agents.status == "active").sum()),
        "mean_final_capital": float(capital.mean()),
        "median_final_capital": float(capital.median()),
        "final_capital_gini": scope.gini(capital),
        "top_decile_mean": float(capital.loc[cohorts["top"]].mean()),
        "bottom_decile_mean": float(capital.loc[cohorts["bot"]].mean()),
    }
    agents.to_csv(output / "agents.csv")
    events.to_csv(output / "events.csv", index=False)
    snapshots.to_csv(output / "agent_daily_snapshots.csv", index=False)
    model.model_history_df().to_csv(output / "model_history.csv", index=False)
    trajectory.to_csv(output / "capital_trajectory.csv")
    counts.to_csv(output / "activity_counts.csv")
    # Directly-follows counts are separate from node/event counts. No omitted
    # activity is bridged: these are adjacency counts in the full lifecycle log.
    paper_counts = {}
    for label, key in [("top", "top"), ("bottom", "bot")]:
        cohort_events = events[events.agent_id.isin(cohorts[key])].copy()
        cohort_events["next_activity"] = cohort_events.groupby("agent_id").activity.shift(-1)
        edges = cohort_events.dropna(subset=["next_activity"]).groupby(
            ["activity", "next_activity"]
        ).size().rename("frequency")
        edges.to_csv(output / f"{label}_dfg_edges.csv")
        grouped_events = cohort_events.groupby("agent_id")
        filtered, _, _, _ = dfg_filtering.filter_dfg_on_activities_percentage(
            edges.to_dict(), grouped_events.activity.first().value_counts().to_dict(),
            grouped_events.activity.last().value_counts().to_dict(),
            cohort_events.activity.value_counts().to_dict(), .3,
        )
        paper_counts[f"{label}_10pct"] = pd.Series(incoming_edge_counts(filtered))
    paper_counts = pd.DataFrame(paper_counts).fillna(0).astype(int).rename_axis("activity")
    paper_counts.to_csv(output / "paper_activity_counts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for key, label, color in [("top", "Top 10%", "#0072B2"), ("mid", "Middle 50%", "#555555"), ("bot", "Bottom 10%", "#D55E00")]:
        mean, std = trajectory[f"{key}_mean"], trajectory[f"{key}_std"]
        axes[0].plot(trajectory.index, mean, label=label, color=color)
        axes[0].fill_between(trajectory.index, mean - std, mean + std, color=color, alpha=.12)
    axes[0].set(xlabel="Day", ylabel="Mean capital", yscale="log", title="(a) Capital trajectories")
    axes[0].set_yticks([1000, 2000, 4000, 8000])
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].legend()
    activities = ["Win reward", "Continue mining", "Join pool", "Leave pool", "Pause mining", "Losing streak"]
    paper_counts.reindex(activities).rename(columns={"top_10pct": "Top 10%", "bottom_10pct": "Bottom 10%"}).plot.barh(ax=axes[1], color=["#0072B2", "#D55E00"])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Retained incoming-edge count", ylabel="", title="(b) Filtered DFG counts")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"baseline_figure2.{extension}", dpi=180)
    plt.close(fig)
    manifest = {
        "parameters": asdict(scope.params), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["mesa", "numpy", "pandas", "matplotlib", "scipy"]},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["notebook.ipynb", "scripts/mining_hardware.py", "scripts/process_policy.py", "scripts/run_baseline.py"]},
        "measurement": "Standard Gini of final capital without shifting values; trajectories from last event per agent/day; activity panel from retained incoming DFG edges. Direct event counts and daily snapshots are also exported.",
        "completed": True,
    }
    (output / "experiment_config.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved baseline tables and Figure 2: {output}")


if __name__ == "__main__":
    main()
