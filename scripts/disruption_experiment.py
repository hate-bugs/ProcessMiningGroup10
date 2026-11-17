"""Paired shutdown and temporary-cost interventions using the notebook model.

Shutdowns are exogenous closures with liquidation, not hardware destruction.
They happen before operating decisions on the intervention day. Forced records
are retained for accounting but excluded from adviser fitting.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.online_feedback_experiment import assigned_agents, bootstrap, metrics, online_protocol
from scripts.process_policy import TransitionProcessAdviser


DEFAULT_SEVERITIES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


@dataclass(frozen=True)
class CostShockPlan:
    start_day: int = 360
    end_day: int = 540  # exclusive: baseline costs resume at the start of this day
    multiplier: float = 2.0

    def __post_init__(self):
        if self.start_day < 2 or self.end_day <= self.start_day:
            raise ValueError("Cost shock needs a pre-intervention day and a positive interval")
        if not np.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("Cost multiplier must be finite and positive")

    def at(self, day):
        return self.multiplier if self.start_day <= day < self.end_day else 1.0


@dataclass(frozen=True)
class ShutdownPlan:
    day: int = 360
    percentage: float = 25
    seed: int = 29101

    def __post_init__(self):
        if self.day < 1 or not 0 <= self.percentage <= 100:
            raise ValueError("Shutdown day must be positive and percentage within [0, 100]")

    def targets(self, model):
        active = {a.unique_id for a in model.agents if a.active}
        # One fixed ranking across severities: larger interventions contain the
        # smaller ones. This stream never consumes simulator randomness.
        ranking = np.random.default_rng(np.random.SeedSequence([self.seed, 73129])).permutation(
            np.arange(1, model.params.n_agents + 1))
        eligible = [int(i) for i in ranking if i in active]
        count = math.ceil(len(eligible) * self.percentage / 100)
        return eligible, set(eligible[:count])


def shutdown_model_class(base_class):
    """Extend the actual notebook model at its existing start-of-day hook."""
    class ShutdownMiningModel(base_class):
        def __init__(self, *args, shutdown_plan=None, cost_shock_plan=None, **kwargs):
            if shutdown_plan is not None and cost_shock_plan is not None:
                raise ValueError("Use one intervention per experiment")
            self.shutdown_plan = shutdown_plan
            self.cost_shock_plan = cost_shock_plan
            self.applied_cost_multiplier = 1.0
            self.shutdown_audit = None
            super().__init__(*args, **kwargs)

        def update_environment(self):
            super().update_environment()
            # The original method redraws each day's grid prices from its own
            # exogenous stream. Multiplication therefore never compounds, and
            # disappears exactly on end_day. Existing profitability, reserve,
            # investment and mine/pause decisions all see the changed prices.
            self.applied_cost_multiplier = self.cost_shock_plan.at(self.steps) if self.cost_shock_plan else 1.0
            if self.applied_cost_multiplier != 1.0:
                self.energy_grids = {grid: price * self.applied_cost_multiplier
                                     for grid, price in self.energy_grids.items()}
            plan = self.shutdown_plan
            if plan is None or self.steps != plan.day:
                return
            eligible, targets = plan.targets(self)
            self.shutdown_audit = dict(
                day=self.steps, eligible=len(eligible), shutdown_count=len(targets),
                requested_pct=plan.percentage,
                actual_pct=100 * len(targets) / len(eligible) if eligible else 0,
                eligible_ids=eligible, shutdown_ids=sorted(targets),
            )
            for agent in self.agents:
                if agent.unique_id not in targets:
                    continue
                before = agent.net_worth()
                cash_before = agent.capital
                context = agent.decision_context()
                state_before = agent.decision_state()
                previous_action = agent.last_action or "START"
                self.log_event(agent, "External shutdown", reason="permanent_disruption")
                agent.exit_market(reason="external_shutdown")
                after = agent.net_worth()
                agent.max_drawdown = max(agent.max_drawdown,
                    (agent.peak_net_worth - after) / max(agent.peak_net_worth, 1.0))
                # A terminal accounting record closes the case. It is not an
                # agent-selected action and must never teach an adviser policy.
                self.decision_log.append(dict(
                    run_id=self.run_id, agent_id=int(agent.unique_id), cycle=int(self.steps),
                    context=context, previous_action=previous_action, action="external_shutdown",
                    externally_forced=True, terminal=True, exit_reason="external_shutdown",
                    state_before=state_before, state_after=agent.decision_state(),
                    net_worth_before=before, net_worth_after=after,
                    initial_net_worth=agent.initial_net_worth,
                    economic_profit=after - before, normalized_reward=(after - before) / agent.initial_net_worth,
                    cash_change=agent.capital - cash_before, mining_profit=0.0,
                    learning_reward=0.0, pm_score=0.0, pm_bonus=0.0, pm_support=0,
                    capital_action="no_review", capital_pm_bonus=0.0,
                ))
    return ShutdownMiningModel


def voluntary_history(model, day, window):
    history = model.decision_log_df()
    history = history[history.cycle > day - window].copy()
    if "externally_forced" in history:
        history = history[~history.externally_forced.eq(True)].copy()
    # The last voluntary event of a forced-shutdown case has no voluntary
    # successor and is censored by TransitionProcessAdviser.fit.
    return history


def disruption_protocol(smoke=False):
    p = online_protocol(smoke)
    return dict(experiment="shutdown", smoke=smoke, n_agents=p["n_agents"],
                days=12 if smoke else 780, shutdown_day=6 if smoke else 360,
                severities=[0, 10, 100] if smoke else DEFAULT_SEVERITIES,
                seeds=[99401, 99402] if smoke else list(range(29101, 29109)),
                feedback_interval=p["feedback_interval"], window_days=p["window_days"],
                min_support=p["min_support"], min_agents=p["min_agents"], shrinkage=p["shrinkage"],
                bootstrap_samples=p["bootstrap_samples"], weight=6.0, discount=0.0,
                learning_rate=0.18, exploration_rate=0.09)


def cost_shock_protocol(smoke=False):
    p = disruption_protocol(smoke)
    p.update(experiment="cost_shock", cost_multipliers=[1.0, 2.0],
             shock_end_day=9 if smoke else 540)
    p.pop("severities")
    return p


def scenario_values(protocol):
    return protocol["cost_multipliers"] if protocol.get("experiment") == "cost_shock" else protocol["severities"]


def baseline_value(protocol):
    return 1.0 if protocol.get("experiment") == "cost_shock" else 0.0


def validate_disruption_protocol(p):
    if not 2 <= p["shutdown_day"] < p["days"]:
        raise ValueError("Intervention must have a preceding reference day and follow-up")
    if p["n_agents"] < 4 or p["n_agents"] % 2:
        raise ValueError("Use an even population of at least four")
    if len(set(p["seeds"])) != len(p["seeds"]) or len(p["seeds"]) < 2:
        raise ValueError("Use at least two distinct independent seeds")
    values = scenario_values(p)
    if not values or len(set(values)) != len(values) or baseline_value(p) not in values:
        raise ValueError("Scenarios must be unique and include the no-intervention baseline")
    if p.get("experiment") == "cost_shock":
        if p["shock_end_day"] > p["days"]:
            raise ValueError("Include a recovery phase after the cost shock")
        for value in values:
            CostShockPlan(p["shutdown_day"], p["shock_end_day"], value)
    else:
        for value in values:
            ShutdownPlan(p["shutdown_day"], value)


def run_market(base_class, base_params, protocol, seed, percentage, family, swap):
    p = protocol
    cost_shock = p.get("experiment") == "cost_shock"
    params = replace(base_params, n_agents=p["n_agents"], cycles=p["days"], seed=seed,
                     daily_reward_coins=base_params.daily_reward_coins * p["n_agents"] / base_params.n_agents,
                     cost_per_hash_factor=6.2, energy_shock_prob=.14,
                     energy_shock_multiplier_min=2.5, energy_shock_multiplier_max=4.5,
                     learning_rate=p["learning_rate"], exploration_rate=p["exploration_rate"], pm_weight=0.)
    model_class = shutdown_model_class(base_class)
    run_id = f"{'cost-shock' if cost_shock else 'shutdown'}-{percentage:g}-{family}-{seed}-swap{int(swap)}"
    intervention = dict(cost_shock_plan=CostShockPlan(p["shutdown_day"], p["shock_end_day"], percentage)) if cost_shock else dict(
        shutdown_plan=ShutdownPlan(p["shutdown_day"], percentage, seed))
    model = model_class(params, run_id=run_id, **intervention)
    treated = assigned_agents(p["n_agents"], seed, swap) if family == "process" else set()
    for agent in model.agents:
        agent._pm_weight = p["weight"] if agent.unique_id in treated else 0.
    meta = dict(seed=seed, scenario_value=percentage, family=family, swap=int(swap), run_id=run_id)
    meta["cost_multiplier" if cost_shock else "severity_pct"] = percentage
    checkpoints = {1, p["shutdown_day"] - 1, p["shutdown_day"], p["shutdown_day"] + 1, p["days"]}
    if cost_shock:
        checkpoints.update([p["shock_end_day"] - 1, p["shock_end_day"], p["shock_end_day"] + 1])
    snapshots, fits, cohort_daily = [], [], []
    target_ids = set()
    original_agents = list(model.agents)
    original_ids = {a.unique_id for a in original_agents}
    cohorts = {"market": original_agents}
    if family == "process":
        cohorts.update(pm=[a for a in original_agents if a.unique_id in treated],
                       control=[a for a in original_agents if a.unique_id not in treated])
    reference_wealth = None
    for day in range(1, p["days"] + 1):
        before = {a.unique_id: (a.total_rewards, a.total_costs, a.times_mined) for a in original_agents}
        decision_start = len(model.decision_log)
        model.step()
        decisions = {r["agent_id"]: r for r in model.decision_log[decision_start:] if not r.get("externally_forced", False)}
        wealth = {a.unique_id: a.net_worth() for a in original_agents}
        if day == p["shutdown_day"] - 1:
            reference_wealth = wealth.copy()
        if model.shutdown_audit:
            target_ids = set(model.shutdown_audit["shutdown_ids"])
        for cohort, members in cohorts.items():
            records = [decisions[a.unique_id] for a in members if a.unique_id in decisions]
            rewards = sum(a.total_rewards - before[a.unique_id][0] for a in members)
            costs = sum(a.total_costs - before[a.unique_id][1] for a in members)
            active_count = sum(a.active for a in members)
            cohort_daily.append(dict(**meta, day=day, cohort=cohort, initial_agents=len(members),
                active_agents=active_count,
                mining_participants=sum(a.times_mined > before[a.unique_id][2] for a in members),
                decisions=len(records),
                **{f"action_{action}": sum(r["action"] == action for r in records)
                   for action in ["mine", "pause", "repair", "join_pool", "leave_pool", "exit"]},
                **{f"capital_{action}": sum(r.get("capital_action") == action for r in records)
                   for action in ["invest", "sell_hardware", "hold_hardware"]},
                operating_cost=costs, mining_reward=rewards, net_mining_profit=rewards - costs,
                normalized_operating_profit_pp=100 * np.mean([
                    ((a.total_rewards - before[a.unique_id][0]) - (a.total_costs - before[a.unique_id][1])) / a.initial_net_worth
                    for a in members]),
                mean_net_worth=float(np.mean([wealth[a.unique_id] for a in members])),
                mean_return_pct=100 * np.mean([(wealth[a.unique_id] - a.initial_net_worth) / a.initial_net_worth for a in members]),
                applied_cost_multiplier=model.applied_cost_multiplier))
            if cohort == "market":
                model.model_history[-1]["actual_mining_participants"] = cohort_daily[-1]["mining_participants"]
                model.model_history[-1]["deployed_hash"] = model.current_total_mining_hash
        if day % p["feedback_interval"] == 0 or day in checkpoints:
            agents = model.agent_summary_df()
            subsets = {"market": agents}
            if family == "process":
                subsets.update(pm=agents[agents.agent_id.isin(treated)],
                               control=agents[~agents.agent_id.isin(treated)])
            for cohort, subset in subsets.items():
                if reference_wealth is not None:
                    denominator = subset.agent_id.map(reference_wealth)
                    valid_reference = bool((denominator > 0).all())
                    secondary = float(100 * ((subset.net_worth - denominator) / denominator).mean()) if valid_reference else np.nan
                else:
                    secondary, valid_reference = np.nan, False
                snapshots.append(dict(**meta, day=day, cohort=cohort, **metrics(subset),
                    forced_shutdown_pct=100 * subset.exit_reason.eq("external_shutdown").mean(),
                    return_relative_reference_wealth_pct=secondary, valid_reference_wealth=valid_reference))
        if day % p["feedback_interval"] == 0 and day < p["days"] and family == "process":
            history = voluntary_history(model, day, p["window_days"])
            adviser = TransitionProcessAdviser(discount=p["discount"], min_support=p["min_support"],
                min_agents=p["min_agents"], shrinkage=p["shrinkage"]).fit(history)
            model.process_adviser = adviser
            fits.append(dict(**meta, update_day=day, first_affected_day=day + 1,
                max_observed_day=int(history.cycle.max()) if len(history) else 0,
                decisions=len(history), forced_records_used=int(history.get("externally_forced", pd.Series(dtype=bool)).eq(True).sum())))
    model.finalize_run()
    agents = model.agent_summary_df().assign(**meta)
    agents["cohort"] = np.where(agents.agent_id.isin(treated), "pm", "control")
    agents["externally_shutdown"] = agents.agent_id.isin(target_ids)
    if {a.unique_id for a in model.agents} != original_ids:
        raise ValueError("This fixed-cohort experiment does not support entrants or removed identities")
    audit = dict(**meta, **(model.shutdown_audit or dict(day=p["shutdown_day"],
        shock_end_day=p.get("shock_end_day"), eligible=None, shutdown_count=0, shutdown_ids=[])))
    audit["pm_shutdown_count"] = len(treated & target_ids)
    audit["control_shutdown_count"] = len(target_ids - treated)
    audit["params"] = asdict(params)
    # Daily market data come from the existing model's end-of-day collection.
    daily = model.model_history_df().assign(**meta)
    events = model.event_log_df()
    events = events[events.reason.isin(["external_shutdown", "permanent_disruption"])].assign(**meta)
    return dict(snapshots=pd.DataFrame(snapshots), agents=agents, audit=audit, cohort_daily=pd.DataFrame(cohort_daily),
                daily=daily, events=events, fits=pd.DataFrame(fits))


def run_job(job):
    # Each worker loads the simulation definitions from the notebook.
    import contextlib
    import io
    from scripts.run_feedback_experiment import load_model
    with contextlib.redirect_stdout(io.StringIO()):
        _, scope = load_model()
    p, seed, percentage, family, swap = job
    return run_market(scope.AdaptiveMiningModel, scope.params, p, seed, percentage, family, swap)


def summarize(snapshots, protocol):
    final = snapshots[snapshots.day == protocol["days"]]
    # Complementary assignments are averaged before resampling seeds.
    means = final.groupby(["seed", "severity_pct", "family", "cohort"])[
        ["mean_return_pct", "survival_pct", "mean_max_drawdown_pct"]].mean()
    rows = []
    for seed in protocol["seeds"]:
        baseline_pm = means.loc[(seed, 0, "process", "pm"), "mean_return_pct"]
        baseline_control = means.loc[(seed, 0, "process", "control"), "mean_return_pct"]
        baseline_market = means.loc[(seed, 0, "process", "market"), "mean_return_pct"]
        for pct in protocol["severities"]:
            pm = means.loc[(seed, pct, "process", "pm")]
            control = means.loc[(seed, pct, "process", "control")]
            market = means.loc[(seed, pct, "process", "market")]
            unguided = means.loc[(seed, pct, "all_control", "market")]
            rows.append(dict(seed=seed, severity_pct=pct,
                pm_return=pm.mean_return_pct, control_return=control.mean_return_pct,
                pm_advantage=pm.mean_return_pct - control.mean_return_pct,
                advantage_change_vs_no_shutdown=pm.mean_return_pct - control.mean_return_pct - baseline_pm + baseline_control,
                market_return=market.mean_return_pct,
                market_change_vs_no_shutdown=market.mean_return_pct - baseline_market,
                market_vs_unguided=market.mean_return_pct - unguided.mean_return_pct,
                survival_pct=market.survival_pct,
                survival_change_vs_no_shutdown=market.survival_pct - means.loc[(seed, 0, "process", "market"), "survival_pct"],
                mean_max_drawdown_pct=market.mean_max_drawdown_pct))
    seeds = pd.DataFrame(rows)
    effects = []
    for pct, group in seeds.groupby("severity_pct"):
        for metric in seeds.columns.difference(["seed", "severity_pct"]):
            effects.append(dict(severity_pct=pct, metric=metric,
                **bootstrap(group[metric], protocol["bootstrap_samples"])))
    return seeds, pd.DataFrame(effects)


def export_results(results, protocol, output):
    output = Path(output)
    frames = {}
    for key in ["snapshots", "agents", "daily", "events", "fits", "cohort_daily"]:
        parts = [r[key] for r in results if not r[key].empty]
        frames[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        frames[key].to_csv(output / f"{key}.csv", index=False)
    audits = [r["audit"] for r in results]
    audit_name = "intervention_audit.json" if protocol.get("experiment") == "cost_shock" else "shutdown_audit.json"
    (output / audit_name).write_text(json.dumps(audits, indent=2), encoding="utf-8")
    if protocol.get("experiment") != "cost_shock":
        seeds, effects = summarize(frames["snapshots"], protocol)
        seeds.to_csv(output / "seed_effects.csv", index=False)
        effects.to_csv(output / "effects.csv", index=False)
        make_figure(effects, output, protocol)
        write_report(effects, audits, protocol, output)
        # Retain the historical separate-cohort figure/CSV schema in new runs.
        from scripts.plot_disruption_cohorts import main as plot_cohorts
        plot_cohorts(["--input", str(output / "snapshots.csv"), "--output", str(output),
                      "--shutdown-day", str(protocol["shutdown_day"])])
    from scripts.robustness_analysis import export_robustness_analysis
    return export_robustness_analysis(frames, protocol, output)


def make_figure(effects, output, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes,
        ["pm_advantage", "advantage_change_vs_no_shutdown", "market_change_vs_no_shutdown"],
        ["Guided minus concurrent controls", "Change in guided advantage", "Change in mean market return"]):
        data = effects[effects.metric == metric].sort_values("severity_pct")
        ax.plot(data.severity_pct, data.mean_delta, marker="o", color="#0072B2")
        ax.fill_between(data.severity_pct, data.ci_low, data.ci_high, alpha=.18, color="#0072B2")
        ax.axhline(0, color="gray", linestyle="--", linewidth=.8)
        ax.set(title=title, xlabel="Requested shutdown (% of active miners)", ylabel="Percentage points")
    fig.suptitle("SMOKE TEST — not report results" if protocol["smoke"] else
                 f"Permanent shutdowns on day {protocol['shutdown_day']}; outcomes at day {protocol['days']}")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"shutdown_sweep.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(effects, audits, protocol, output):
    def estimate(pct, metric):
        row = effects[(effects.severity_pct == pct) & (effects.metric == metric)].iloc[0]
        return f"{row.mean_delta:+.2f} [{row.ci_low:+.2f}, {row.ci_high:+.2f}]"
    lines = ["# Permanent-shutdown robustness experiment", "",
        "**SMOKE TEST ONLY — not report results.**" if protocol["smoke"] else "Fresh supplementary experiment; settings frozen before this sweep.", "",
        f"At the start of day {protocol['shutdown_day']}, a randomly selected fraction of currently active miners "
        "is permanently closed. Installed hardware is liquidated and outstanding orders are cancelled using the "
        "existing model's refund policy. Closed miners receive no further mining rewards, costs, or decisions. "
        "This is an operational closure with asset recovery, not destruction or confiscation of assets.", "",
        f"Each severity uses {len(protocol['seeds'])} new seeds, a 50%-guided market with complementary assignments, "
        "and an all-control market. The full adviser's weight 6 and discount 0 are retained without retuning. "
        "Larger severities use nested random target sets within each market. Forced closures are excluded from "
        "policy learning. Returns include all initially assigned miners, including earlier exits and forced closures.", "",
        "The denominator is the active population immediately before the intervention, not the original population. "
        "Positive fractional headcounts round upward. Small adjacent percentages can therefore close the same miners. "
        "An all-control market may have a different active population, so equal percentages need not imply equal headcounts.", "",
        "| Requested % | Mean closed / eligible in mixed markets | Guided advantage, pp [95% interval] | Change in advantage vs 0%, pp | Market return change vs 0%, pp |",
        "|---:|---:|---:|---:|---:|"]
    for pct in protocol["severities"]:
        selected = [a for a in audits if a["severity_pct"] == pct and a["family"] == "process"]
        closed = np.mean([a["shutdown_count"] for a in selected])
        eligible = np.mean([a["eligible"] for a in selected])
        lines.append(f"| {pct:g} | {closed:.2f} / {eligible:.2f} | {estimate(pct, 'pm_advantage')} | "
                     f"{estimate(pct, 'advantage_change_vs_no_shutdown')} | {estimate(pct, 'market_change_vs_no_shutdown')} |")
    lines += ["", "Intervals use seed-level paired bootstrap resampling after averaging complementary assignments. "
        "All severity comparisons are exploratory, correlated, and unadjusted for multiple comparisons. "
        "A positive final guided advantage can include gains accumulated before shutdown; robustness of the "
        "advantage to disruption must also be judged from its paired change relative to the 0% run. "
        "Do not interpret non-significance as equivalence or resilience. At 100%, all remaining miners close "
        "and subsequent network production is zero; final positive wealth can still reflect earlier gains and liquidation.", "",
        "This extension tests the previously selected full adviser. It does not establish an additional benefit "
        "from sequence information, repeated refitting, or greater market-wide welfare.", "",
        "![Shutdown severity sweep](shutdown_sweep.png)", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def manifest(protocol, completed=False):
    names = ["notebook.ipynb", "scripts/mining_hardware.py", "scripts/process_policy.py", "scripts/online_feedback_experiment.py",
             "scripts/disruption_experiment.py", "scripts/robustness_analysis.py", "scripts/run_disruption_experiment.py",
             "scripts/plot_disruption_cohorts.py"]
    return dict(schema_version=2, protocol=protocol, completed=completed, simulation_rerun=True,
        design="Paired intervention study; fixed cohorts; no retuning",
        aggregation="Average complementary assignments within each seed, then resample independent seeds",
        bootstrap=dict(samples=protocol["bootstrap_samples"], random_seed=19431, interval="95% percentile; pointwise; unadjusted"),
        reference_day=protocol["shutdown_day"] - 1,
        primary_metric="D(t) = [P_guided(t)-P_control(t)]_intervention - [P_guided(t)-P_control(t)]_baseline",
        packages={n: importlib.metadata.version(n) for n in ["mesa", "numpy", "pandas", "scipy", "matplotlib"]},
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names})
