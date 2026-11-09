"""Shared seed-level post-intervention, behavior, and recovery analysis.

Integrated into disruption_experiment.export_results for both interventions.
Days and complementary assignments are never bootstrap replication units.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from online_feedback_experiment import bootstrap


def seed_statistics(data, groups, metrics, samples):
    rows = []
    for key, group in data.groupby(groups, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        if group.seed.duplicated().any():
            raise ValueError("Aggregate dependent records within each seed before bootstrap")
        for metric in metrics:
            if group[metric].isna().any():
                # Secondary reference-wealth returns are unavailable when any
                # original cohort member has nonpositive reference wealth.
                if metric == "return_relative_reference_wealth_pct":
                    continue
                raise ValueError(f"Missing seed-level observations for {metric}")
            result = bootstrap(group[metric], samples)
            result["mean"] = result.pop("mean_delta")
            rows.append(dict(zip(groups, key), metric=metric, **result))
    return pd.DataFrame(rows)


def post_intervention_tables(snapshots, protocol):
    keys = ["seed", "scenario_value", "family", "cohort"]
    metrics = ["mean_return_pct", "survival_pct"]
    if "return_relative_reference_wealth_pct" in snapshots:
        metrics.append("return_relative_reference_wealth_pct")
    if snapshots.duplicated(keys + ["swap", "day"]).any():
        raise ValueError("Duplicate cohort checkpoint")
    counts = snapshots.groupby(keys + ["day"]).swap.nunique().reset_index(name="assignments")
    expected = np.where(counts.family.eq("process"), 2, 1)
    if not np.array_equal(counts.assignments, expected):
        raise ValueError("Incomplete complementary assignments")
    seeds = snapshots.groupby(keys + ["day"], as_index=False)[metrics].mean()
    secondary = "return_relative_reference_wealth_pct"
    if secondary in metrics:
        strict = snapshots.groupby(keys + ["day"])[secondary].agg(lambda values: values.mean(skipna=False))
        seeds[secondary] = strict.reindex(pd.MultiIndex.from_frame(seeds[keys + ["day"]])).to_numpy()
    expected_seeds = set(protocol["seeds"])
    for _, group in seeds.groupby(["scenario_value", "family", "cohort", "day"]):
        if set(group.seed) != expected_seeds:
            raise ValueError("Every checkpoint needs the complete configured independent seed set")
    reference = seeds[seeds.day == protocol["shutdown_day"] - 1][keys + ["mean_return_pct"]]
    reference = reference.rename(columns={"mean_return_pct": "reference_return_pct"})
    post = seeds[seeds.day >= protocol["shutdown_day"] - 1].merge(reference, on=keys, validate="many_to_one")
    if len(post) != int((seeds.day >= protocol["shutdown_day"] - 1).sum()):
        raise ValueError("Missing pre-intervention reference for some cohorts")
    post["post_return_pp"] = post.mean_return_pct - post.reference_return_pct
    mixed = post[(post.family == "process") & post.cohort.isin(["pm", "control"])]
    wide = mixed.pivot(index=["seed", "scenario_value", "day"], columns="cohort", values="post_return_pp")
    advantages = (wide.pm - wide.control).rename("post_advantage_pp").reset_index()
    base_value = 1.0 if protocol.get("experiment") == "cost_shock" else 0.0
    base = advantages[advantages.scenario_value == base_value][["seed", "day", "post_advantage_pp"]]
    base = base.rename(columns={"post_advantage_pp": "baseline_post_advantage_pp"})
    paired = advantages.merge(base, on=["seed", "day"], validate="many_to_one")
    if len(paired) != len(advantages):
        raise ValueError("Missing matched baseline checkpoints")
    paired["additional_advantage_pp"] = paired.post_advantage_pp - paired.baseline_post_advantage_pp
    cohort_baseline = post[post.scenario_value == base_value][
        ["seed", "day", "family", "cohort", "post_return_pp"]].rename(columns={"post_return_pp": "baseline_post_return_pp"})
    post = post.merge(cohort_baseline, on=["seed", "day", "family", "cohort"], validate="many_to_one")
    post["post_return_change_vs_baseline_pp"] = post.post_return_pp - post.baseline_post_return_pp
    return seeds, post, paired


def phase_tables(seed_returns, cohort_daily, protocol):
    if protocol.get("experiment") != "cost_shock":
        return pd.DataFrame(), pd.DataFrame()
    start, end, final = protocol["shutdown_day"], protocol["shock_end_day"], protocol["days"]
    keys = ["seed", "scenario_value", "family", "cohort"]
    table = seed_returns.pivot(index=keys, columns="day", values="mean_return_pct")
    phases = []
    for key, row in table.iterrows():
        for name, length, value in [
            ("before", start - 1, row[start - 1]),
            ("during", end - start, row[end - 1] - row[start - 1]),
            ("after", final - end + 1, row[final] - row[end - 1])]:
            phases.append(dict(zip(keys, key), phase=name, phase_return_pp=value,
                               return_pp_per_day=value / length))
    phases = pd.DataFrame(phases)
    daily = cohort_daily.copy()
    daily["phase"] = np.where(daily.day < start, "before", np.where(daily.day < end, "during", "after"))
    phase_behavior = daily.groupby(keys + ["swap", "phase"], as_index=False)[
        ["active_agents", "mining_participants", "operating_cost", "net_mining_profit", "normalized_operating_profit_pp"]].mean()
    phase_behavior = phase_behavior.groupby(keys + ["phase"], as_index=False).mean(numeric_only=True).drop(columns="swap")
    phases = phases.merge(phase_behavior, on=keys + ["phase"], validate="one_to_one")
    # Explicit recovery metric: change in the paired scenario-minus-baseline
    # return gap between end of shock and final day. Positive means the relative
    # position improved; it only closes a deficit if the starting gap was negative.
    recovery = []
    baseline = table.xs(1.0, level="scenario_value")
    for (seed, value, family, cohort), row in table.iterrows():
        base = baseline.loc[(seed, family, cohort)]
        end_gap = row[end - 1] - base[end - 1]
        final_gap = row[final] - base[final]
        recovery.append(dict(seed=seed, scenario_value=value, family=family, cohort=cohort,
            gap_at_shock_end_pp=end_gap, final_gap_pp=final_gap,
            recovery_gap_change_pp=final_gap - end_gap))
    return phases, pd.DataFrame(recovery)


def behavior_tables(cohort_daily, daily, protocol):
    keys = ["seed", "scenario_value", "family", "cohort"]
    data = cohort_daily.copy()
    start = protocol["shutdown_day"]
    end = protocol.get("shock_end_day", protocol["days"] + 1)
    data["phase"] = np.where(data.day < start, "before", np.where(data.day < end, "during", "after"))
    data["bin_start"] = ((data.day - 1) // 30) * 30 + 1
    data["participation_pct"] = 100 * data.mining_participants / data.initial_agents
    data["active_pct"] = 100 * data.active_agents / data.initial_agents
    for action in ["mine", "pause", "join_pool", "leave_pool"]:
        data[f"{action}_per_100_assigned"] = 100 * data[f"action_{action}"] / data.initial_agents
    data["invest_per_100_assigned"] = 100 * data.capital_invest / data.initial_agents
    fields = ["active_agents", "participation_pct", "operating_cost", "net_mining_profit",
              "normalized_operating_profit_pp", "mean_net_worth", "mine_per_100_assigned",
              "pause_per_100_assigned", "join_pool_per_100_assigned", "leave_pool_per_100_assigned", "invest_per_100_assigned"]
    # Bins are split at intervention boundaries so no average straddles a phase.
    bins = data.groupby(keys + ["swap", "phase", "bin_start"], as_index=False)[fields].mean()
    bins = bins.groupby(keys + ["phase", "bin_start"], as_index=False)[fields].mean()
    network = daily[daily.cycle > 0].copy()
    if "actual_mining_participants" in network:
        network["mining_agents"] = network.actual_mining_participants
        network["mining_hash_power"] = network.deployed_hash
    network["bin_start"] = ((network.cycle - 1) // 30) * 30 + 1
    network["phase"] = np.where(network.cycle < start, "before", np.where(network.cycle < end, "during", "after"))
    # Descriptive opportunity among actual participants: changes can reflect
    # competition/composition and must not be attributed to learning alone.
    network["reward_per_participant"] = network.paid_mining_rewards / network.mining_agents.replace(0, np.nan)
    network["reward_per_hash"] = network.paid_mining_rewards / network.mining_hash_power.replace(0, np.nan)
    network_fields = ["active_agents", "mining_agents", "mining_hash_power", "paid_mining_rewards", "reward_per_participant", "reward_per_hash"]
    network = network.groupby(["seed", "scenario_value", "family", "swap", "phase", "bin_start"], as_index=False)[network_fields].mean()
    network = network.groupby(["seed", "scenario_value", "family", "phase", "bin_start"], as_index=False)[network_fields].mean()
    def midpoint(row):
        lower, upper = row.bin_start, min(row.bin_start + 29, protocol["days"])
        if row.phase == "before":
            upper = min(upper, start - 1)
        elif row.phase == "during":
            lower, upper = max(lower, start), min(upper, end - 1)
        else:
            lower = max(lower, end)
        return (lower + upper) / 2
    bins["bin_day"] = bins.apply(midpoint, axis=1)
    network["bin_day"] = network.apply(midpoint, axis=1)
    return bins, network


def export_robustness_analysis(frames, protocol, output):
    output = Path(output)
    samples = protocol["bootstrap_samples"]
    returns, post, paired = post_intervention_tables(frames["snapshots"], protocol)
    phases, recovery = phase_tables(returns, frames["cohort_daily"], protocol)
    behaviors, network = behavior_tables(frames["cohort_daily"], frames["daily"], protocol)
    tables = dict(cohort_return_seed_values=returns, post_intervention_seed_values=post,
                  paired_advantage_seed_values=paired, behavior_seed_values=behaviors,
                  network_seed_values=network)
    tables["cumulative_return_effects"] = seed_statistics(returns,
        ["scenario_value", "family", "cohort", "day"], ["mean_return_pct", "survival_pct"], samples)
    post_metrics = ["post_return_pp", "post_return_change_vs_baseline_pp"]
    if "return_relative_reference_wealth_pct" in post:
        post_metrics.append("return_relative_reference_wealth_pct")
    tables["post_intervention_effects"] = seed_statistics(post,
        ["scenario_value", "family", "cohort", "day"], post_metrics, samples)
    tables["paired_advantage_effects"] = seed_statistics(paired,
        ["scenario_value", "day"], ["post_advantage_pp", "additional_advantage_pp"], samples)
    tables["behavior_effects"] = seed_statistics(behaviors,
        ["scenario_value", "family", "cohort", "phase", "bin_start", "bin_day"],
        ["active_agents", "participation_pct", "normalized_operating_profit_pp", "mine_per_100_assigned",
         "pause_per_100_assigned", "join_pool_per_100_assigned", "leave_pool_per_100_assigned", "invest_per_100_assigned"], samples)
    tables["network_effects"] = seed_statistics(network,
        ["scenario_value", "family", "phase", "bin_start", "bin_day"],
        ["active_agents", "mining_agents", "mining_hash_power", "paid_mining_rewards"], samples)
    # No production means opportunities per participant/hash are undefined,
    # not zero. Retain those NaNs in network_seed_values; plot counts separately.
    if not phases.empty:
        tables.update(phase_seed_values=phases, recovery_seed_values=recovery)
        tables["phase_effects"] = seed_statistics(phases, ["scenario_value", "family", "cohort", "phase"],
            ["phase_return_pp", "return_pp_per_day", "active_agents", "mining_participants",
             "operating_cost", "net_mining_profit", "normalized_operating_profit_pp"], samples)
        tables["recovery_effects"] = seed_statistics(recovery, ["scenario_value", "family", "cohort"],
            ["gap_at_shock_end_pp", "final_gap_pp", "recovery_gap_change_pp"], samples)
        tables["recovery_advantage_seed_values"] = recovery_advantage(recovery)
        tables["recovery_advantage_effects"] = seed_statistics(tables["recovery_advantage_seed_values"],
            ["scenario_value"], ["guided_minus_control_recovery_gap_change_pp"], samples)
    for name, data in tables.items():
        data.to_csv(output / f"{name}.csv", index=False)
    plot_robustness(tables, protocol, output)
    write_robustness_report(tables, protocol, output)
    return tables


def recovery_advantage(recovery):
    data = recovery[(recovery.family == "process") & recovery.cohort.isin(["pm", "control"])]
    wide = data.pivot(index=["seed", "scenario_value"], columns="cohort", values="recovery_gap_change_pp")
    return (wide.pm - wide.control).rename("guided_minus_control_recovery_gap_change_pp").reset_index()


def plot_robustness(tables, protocol, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cost = protocol.get("experiment") == "cost_shock"
    values = protocol["cost_multipliers"] if cost else protocol["severities"]
    colors = {"pm": "#0072B2", "control": "#D55E00"}
    def markers(ax):
        ax.axvline(protocol["shutdown_day"], color="black", linestyle="--", lw=1)
        ax.text(protocol["shutdown_day"] + 3, .97, f"Start {protocol['shutdown_day']}",
                transform=ax.get_xaxis_transform(), va="top", fontsize=7, rotation=90)
        if cost:
            ax.axvline(protocol["shock_end_day"], color="black", linestyle=":", lw=1)
            ax.axvspan(protocol["shutdown_day"], protocol["shock_end_day"], color="gray", alpha=.07)
            ax.text(protocol["shock_end_day"] + 3, .97, f"End {protocol['shock_end_day']}",
                    transform=ax.get_xaxis_transform(), va="top", fontsize=7, rotation=90)
        ax.axhline(0, color="gray", lw=.6)
        ax.grid(alpha=.15)
    def save(fig, name):
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    for table_name, metric, name, ylabel in [
        ("cumulative_return_effects", "mean_return_pct", "cohort_cumulative_returns", "Cumulative return (%)"),
        ("post_intervention_effects", "post_return_pp", "cohort_post_intervention_returns", "Return since reference day (pp of initial wealth)")]:
        columns = min(3, len(values))
        rows = (len(values) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.4 * rows), sharex=True, sharey=True, squeeze=False)
        for ax, value in zip(axes.flat, values):
            for cohort, color in colors.items():
                data = tables[table_name]
                data = data[(data.family == "process") & (data.scenario_value == value) & (data.cohort == cohort) & (data.metric == metric)].sort_values("day")
                ax.plot(data.day, data["mean"], color=color, label="Guided" if cohort == "pm" else "Concurrent unguided")
                ax.fill_between(data.day, data.ci_low, data.ci_high, color=color, alpha=.12)
            markers(ax)
            ax.set(title=f"Cost multiplier {value:g}x" if cost else f"Shutdown {value:g}%", xlabel="Day", ylabel=ylabel)
        for ax in list(axes.flat)[len(values):]:
            ax.set_visible(False)
        axes[0, 0].legend(fontsize=8)
        save(fig, name)
    if cost:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, title in zip(axes, ["post_advantage_pp", "additional_advantage_pp"],
                                     ["Guided advantage since reference day", "Paired change versus baseline (primary)"]):
            for value in values:
                data = tables["paired_advantage_effects"]
                data = data[(data.scenario_value == value) & (data.metric == metric)].sort_values("day")
                ax.plot(data.day, data["mean"], label=f"{value:g}x")
                ax.fill_between(data.day, data.ci_low, data.ci_high, alpha=.12)
            markers(ax)
            ax.set(title=title, xlabel="Day", ylabel="Percentage points")
            ax.legend()
        save(fig, "cost_shock_guided_advantage")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        data = tables["paired_advantage_effects"]
        data = data[(data.day == protocol["days"]) & (data.metric == "additional_advantage_pp")].sort_values("scenario_value")
        ax.plot(data.scenario_value, data["mean"], marker="o")
        ax.fill_between(data.scenario_value, data.ci_low, data.ci_high, alpha=.16)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set(xlabel="Shutdown (% of active miners)", ylabel="Paired change in post-intervention advantage (pp)",
               title="Shutdown effect on the guided advantage: seed-paired comparison")
        save(fig, "shutdown_paired_post_advantage")
    # Show all scenarios for network counts; selected ones for action trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    network = tables["network_seed_values"]
    for value in values:
        data = network[(network.family == "process") & (network.scenario_value == value)]
        data = data.groupby(["bin_day", "phase"])[["active_agents", "reward_per_participant"]].mean().reset_index().sort_values("bin_day")
        for ax, metric in zip(axes, ["active_agents", "reward_per_participant"]):
            ax.plot(data.bin_day, data[metric], label=f"{value:g}{'x' if cost else '%'}")
            markers(ax)
    axes[0].set(title="Active miners in mixed markets", xlabel="Day (30-day bins)", ylabel="Miners")
    axes[1].set(title="Paid rewards per mining participant", xlabel="Day (30-day bins)", ylabel="Scenario currency units per day")
    axes[0].legend(ncol=3, fontsize=7)
    save(fig, "network_participation_and_opportunity")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    focus = values if cost else [v for v in [0, 10, 50, 100] if v in values]
    for ax, metric, title in zip(axes.flat,
        ["participation_pct", "pause_per_100_assigned", "invest_per_100_assigned", "normalized_operating_profit_pp"],
        ["Mining participation (% of original cohort)", "Pause decisions per 100 assigned miners/day",
         "Investment decisions per 100 assigned miners/day", "Daily operating profit (pp of initial wealth)"]):
        for index, value in enumerate(focus):
            for cohort, color in colors.items():
                data = tables["behavior_effects"]
                data = data[(data.family == "process") & (data.scenario_value == value) & (data.cohort == cohort) & (data.metric == metric)].sort_values("bin_day")
                ax.plot(data.bin_day, data["mean"], color=color, linestyle=["-", "--", ":", "-."][index % 4],
                        label=f"{'Guided' if cohort == 'pm' else 'Unguided'} {value:g}{'x' if cost else '%'}")
        markers(ax)
        ax.set(title=title, xlabel="Day (phase-separated 30-day bins)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    save(fig, "cohort_adaptation_trajectories")


def write_robustness_report(tables, protocol, output):
    cost = protocol.get("experiment") == "cost_shock"
    final = protocol["days"]
    values = protocol["cost_multipliers"] if cost else protocol["severities"]
    def estimate(table, **filters):
        data = tables[table]
        for key, value in filters.items():
            data = data[data[key] == value]
        row = data.iloc[0]
        return f"{row['mean']:+.2f} [{row.ci_low:+.2f}, {row.ci_high:+.2f}]"
    lines = [f"# {'Temporary operating-cost shock' if cost else 'Permanent shutdown'}: robustness analysis v2", "",
        "**SMOKE TEST ONLY.**" if protocol["smoke"] else "Fresh simulations with fixed adviser settings and the existing eight disruption-study seeds.", "",
        f"Intervention day: {protocol['shutdown_day']}; reference: end of day {protocol['shutdown_day'] - 1}; horizon: {final}. "
        f"Seeds: {protocol['seeds']}. Complementary assignments are averaged within seed before "
        f"{protocol['bootstrap_samples']:,} seed bootstrap resamples. Intervals are descriptive, pointwise and unadjusted.", "",
        "P(t) = R(t) - R(reference), in percentage points of initial wealth. A(t) = P_guided(t) - P_control(t). "
        "The primary paired comparison is D(t) = A_intervention(t) - A_baseline(t). "
        "All initially assigned miners remain in cohort means; this avoids selecting only survivors. "
        "Secondary returns relative to each miner's reference-day wealth are exported separately and use a different denominator.", ""]
    if cost:
        lines += [f"The energy-price multiplier acts on days {protocol['shutdown_day']} through {protocol['shock_end_day'] - 1}. "
            f"Baseline prices resume at the start of day {protocol['shock_end_day']}. It changes actual energy costs, expected profitability, "
            "cash reserves and investment estimates through existing model calculations. No new actions, forecasts, guidance rules or fixed charges are introduced. "
            "The model's existing daily random energy shocks remain in place; this intervention multiplies the resulting prices.", ""]
    else:
        lines += ["Shutdown selection, liquidation and ceiling rounding are unchanged. Only active miners are eligible; "
            "closed miners never return. There is no entry mechanism in this fixed-population model. "
            "At 100%, production stops; a remaining cumulative advantage is not post-intervention adaptation.", ""]
    lines += ["| Scenario | Guided post return, pp [95% CI] | Unguided post return, pp [95% CI] | Post guided advantage A | Paired change D (primary) |",
              "|---:|---:|---:|---:|---:|"]
    for value in values:
        pg = estimate("post_intervention_effects", scenario_value=value, family="process", cohort="pm", day=final, metric="post_return_pp")
        pu = estimate("post_intervention_effects", scenario_value=value, family="process", cohort="control", day=final, metric="post_return_pp")
        a = estimate("paired_advantage_effects", scenario_value=value, day=final, metric="post_advantage_pp")
        d = estimate("paired_advantage_effects", scenario_value=value, day=final, metric="additional_advantage_pp")
        lines.append(f"| {value:g}{'x' if cost else '%'} | {pg} | {pu} | {a} | {d} |")
    if cost:
        lines += ["", "## Adaptation and recovery", "",
            "Recovery is evaluated explicitly as the change in the paired scenario-minus-baseline return gap "
            "between the last shock day and the final day. A positive change improves the relative position; "
            "it closes a deficit only if the gap at shock end was negative. It is not a measured time-to-recovery. "
            "Phase returns, participation, action frequencies and operating profits are exported for before, during and after the shock.", "",
            "| Scenario / cohort | Gap at shock end | Gap at final day | Gap change after costs restore |", "|---|---:|---:|---:|"]
        for value in values:
            if value == 1:
                continue
            for cohort in ["pm", "control"]:
                cols = [estimate("recovery_effects", scenario_value=value, family="process", cohort=cohort, metric=m)
                        for m in ["gap_at_shock_end_pp", "final_gap_pp", "recovery_gap_change_pp"]]
                lines.append(f"| {value:g}x / {'guided' if cohort == 'pm' else 'unguided'} | " + " | ".join(cols) + " |")
        lines += ["", "The paired guided-minus-control difference in recovery-gap improvement is:", ""]
        for value in values:
            if value != 1:
                lines.append(f"- {value:g}x: " + estimate("recovery_advantage_effects", scenario_value=value,
                    metric="guided_minus_control_recovery_gap_change_pp") + " percentage points.")
        lines += ["", "An interval spanning zero does not establish that guided miners recover better than controls."]
    lines += ["", "## Interpretation limits", "",
        "Paired changes measure the disruption's effect on the existing guided-versus-unguided advantage under "
        "the model, not a pure effect of learning speed. Guidance, endogenous exits, action changes and competition "
        "interact. Per-assigned-miner action counts and active populations are exported alongside returns; "
        "reward opportunities per participant describe the changing competition. A frozen-policy or no-learning "
        "ablation would be needed to isolate adaptation causally. Do not infer significance from marginal-interval overlap.", "",
        "No retuning was performed. Synthetic fixed expected issuance, asset liquidation, no entry and omitted network "
        "security/price feedbacks limit transfer to real mining systems. Inactive miners retain the original cost treatment.", "",
        "## Figures and data", "",
        "- `cohort_cumulative_returns.png/.pdf`: cumulative cohort returns with seed-bootstrap intervals.",
        "- `cohort_post_intervention_returns.png/.pdf`: post-reference returns.",
        f"- `{'cost_shock_guided_advantage' if cost else 'shutdown_paired_post_advantage'}.png/.pdf`: paired guided advantage.",
        "- `network_participation_and_opportunity.png/.pdf`: counts and reward opportunity.",
        "- `cohort_adaptation_trajectories.png/.pdf`: participation, pauses, investments and operating profitability.",
        "- `*_seed_values.csv`: independent-seed inputs; `*_effects.csv`: estimates, intervals and positive-seed counts.",
        "- `cohort_daily.csv`: exact daily behavior, costs, profits and wealth, before aggregation.",
        "- `experiment_config.json`: protocol, source hashes, versions and rerun provenance.", ""]
    (output / "ROBUSTNESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
