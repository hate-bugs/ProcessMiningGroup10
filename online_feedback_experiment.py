"""Paired, randomized partial-adoption experiment with causal-time PM feedback.

The same agents continue throughout each market. Treatment assignment changes
only access to automatically discovered operating-policy scores.
"""
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd

from process_policy import TransitionProcessAdviser


METRICS = ["mean_return_pct", "median_return_pct", "mean_max_drawdown_pct", "survival_pct"]
LOG_COLUMNS = ["run_id", "agent_id", "cycle", "context", "previous_action", "action",
               "terminal", "learning_reward", "normalized_reward", "pm_bonus", "pm_support"]


def online_protocol(smoke=False):
    return {
        "smoke": smoke, "n_agents": 6 if smoke else 40,
        "feedback_interval": 4 if smoke else 60,
        "feedback_updates": 2 if smoke else 12,
        "window_days": 8 if smoke else 180,
        "min_support": 2 if smoke else 5, "min_agents": 2, "shrinkage": 10.0,
        "validation_seeds": [98301, 98302] if smoke else [8301, 8302, 8303],
        "test_seeds": [99301, 99302] if smoke else list(range(19101, 19109)),
        "candidates": [dict(weight=w, discount=g) for w, g in product(
            [2.0] if smoke else [2.0, 6.0], [0.0] if smoke else [0.0, 0.9])],
        "bootstrap_samples": 200 if smoke else 5000,
        "learning_rate": 0.18, "exploration_rate": 0.09,
    }


def validate_protocol(p):
    if p["n_agents"] < 4 or p["n_agents"] % 2:
        raise ValueError("Balanced treatment needs an even population of at least four")
    if p["feedback_updates"] < 1 or p["feedback_interval"] < 2 or p["window_days"] < 2:
        raise ValueError("Feedback needs positive durations and complete transitions")
    validation, test = p["validation_seeds"], p["test_seeds"]
    if min(len(validation), len(test)) < 2 or len(set(validation + test)) != len(validation + test):
        raise ValueError("Use distinct validation and test seeds, with at least two each")


def assigned_agents(n_agents, seed, swap):
    ids = set(range(1, n_agents + 1))
    selected = set(np.random.default_rng(np.random.SeedSequence([seed, 99821])).choice(
        np.arange(1, n_agents + 1), n_agents // 2, replace=False).tolist())
    return ids - selected if swap else selected


def metrics(agents):
    return {
        "mean_return_pct": float(100 * agents.normalized_return.mean()),
        "median_return_pct": float(100 * agents.normalized_return.median()),
        "mean_max_drawdown_pct": float(100 * agents.max_drawdown.mean()),
        "survival_pct": float(100 * agents.status.eq("active").mean()),
    }


def bootstrap(values, samples=5000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite seed-level effects required")
    rng = np.random.default_rng(19431)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return dict(mean_delta=float(values.mean()), ci_low=float(low), ci_high=float(high),
                paired_seeds=len(values), positive_seeds=int((values > 0).sum()))


class OnlineFeedbackExperiment:
    def __init__(self, model_class, params, protocol, output):
        validate_protocol(protocol)
        self.model_class = model_class
        self.protocol = json.loads(json.dumps(protocol))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        p = self.protocol
        self.days = p["feedback_interval"] * (p["feedback_updates"] + 1)
        self.params = replace(params, n_agents=p["n_agents"], cycles=self.days,
                              daily_reward_coins=params.daily_reward_coins * p["n_agents"] / params.n_agents,
                              cost_per_hash_factor=6.2, energy_shock_prob=0.14,
                              energy_shock_multiplier_min=2.5, energy_shock_multiplier_max=4.5,
                              learning_rate=p["learning_rate"], exploration_rate=p["exploration_rate"], pm_weight=0.0)
        self.selection = {}
        self.audit = []
        self.evaluated = False
        self.manifest = {
            "protocol": p, "params": asdict(self.params),
            "primary_endpoint": "PM minus concurrent controls, mean normalized economic return at final day; complementary assignments averaged within each seed",
            "selection_rule": "Maximum average validation-seed treatment difference; deterministic candidate order breaks ties",
            "scope": "Relative competitive advantage at 50% adoption; not a claim about system-wide welfare or all miners",
            "development_seeds": [8101, 8102, 8103, 8201, 8202, 8203],
            "development_note": "Population-wide adoption probes did not improve aggregate outcomes. The partial-adoption design tests a different, explicitly stated question. Prior negative study retained separately.",
            "baseline_parameters": "Learning rate 0.18 and exploration 0.09 retained from the earlier baseline parameter search; identical for both cohorts",
            "packages": {n: importlib.metadata.version(n) for n in ["mesa", "numpy", "pandas", "scipy", "matplotlib"]},
        }
        self._manifest()

    def _manifest(self):
        (self.output / "experiment_config.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def run_market(self, seed, family, candidate, swap=False, stage="validation", record=False):
        p = self.protocol
        treated = assigned_agents(p["n_agents"], seed, swap) if family != "all_control" else set()
        run_id = f"{stage}-{family}-{seed}-swap{int(swap)}"
        model = self.model_class(replace(self.params, seed=seed), run_id=run_id)
        for agent in model.agents:
            agent._pm_weight = candidate["weight"] if agent.unique_id in treated else 0.0
        self.audit.append(dict(stage=stage, family=family, seed=seed, swap=int(swap), **candidate))
        snapshots, edges, transitions, fits = [], [], [], []
        for day in range(1, self.days + 1):
            model.step()
            if day % p["feedback_interval"]:
                continue
            agents = model.agent_summary_df()
            cohorts = {"all_control": agents} if family == "all_control" else {
                "pm": agents[agents.agent_id.isin(treated)], "control": agents[~agents.agent_id.isin(treated)]}
            for cohort, subset in cohorts.items():
                snapshots.append(dict(seed=seed, family=family, swap=int(swap), day=day,
                                      feedback_round=day // p["feedback_interval"] - 1,
                                      cohort=cohort, **metrics(subset)))
            # Measure first, then fit: update at day d can affect only day d+1.
            if day == self.days or family == "all_control":
                continue
            history = model.decision_log_df()[LOG_COLUMNS]
            history = history[history.cycle > day - p["window_days"]]
            adviser = TransitionProcessAdviser(
                discount=candidate["discount"], min_support=p["min_support"],
                min_agents=p["min_agents"], shrinkage=p["shrinkage"],
                use_sequence=family != "no_sequence",
            ).fit(history)
            model.process_adviser = adviser
            fits.append(dict(run_id=run_id, seed=seed, family=family, swap=int(swap),
                             update_day=day, first_affected_day=day + 1,
                             max_observed_day=int(history.cycle.max()) if len(history) else 0,
                             logged_decisions=len(history), supported_edges=int(adviser.table.supported.sum()) if len(adviser.table) else 0))
            if record and len(adviser.table):
                edges.append(adviser.table.assign(update_day=day))
                transitions.append(adviser.transition_table.assign(update_day=day))
        model.finalize_run()
        final_agents = model.agent_summary_df().assign(seed=seed, family=family, swap=int(swap))
        final_agents["cohort"] = np.where(final_agents.agent_id.isin(treated), "pm", "control")
        if record:
            directory = self.output / "evidence"
            directory.mkdir(exist_ok=True)
            log = model.decision_log_df()[LOG_COLUMNS]
            log["treated"] = log.agent_id.isin(treated)
            log.to_json(directory / f"{run_id}_decisions.jsonl.gz", orient="records", lines=True, compression="gzip")
            if edges:
                pd.concat(edges, ignore_index=True).to_csv(directory / f"{run_id}_edges.csv.gz", index=False)
                pd.concat(transitions, ignore_index=True).to_csv(directory / f"{run_id}_transitions.csv.gz", index=False)
        return pd.DataFrame(snapshots), final_agents, pd.DataFrame(fits)

    def tune(self):
        if self.selection:
            raise RuntimeError("Parameter selection already completed")
        rows = []
        for family in ["process", "no_sequence"]:
            for index, candidate in enumerate(self.protocol["candidates"]):
                for seed in self.protocol["validation_seeds"]:
                    for swap in [False, True]:
                        snapshots, _, _ = self.run_market(seed, family, candidate, swap)
                        final = snapshots[snapshots.day == self.days].set_index("cohort")
                        rows.append(dict(family=family, candidate=index, seed=seed, swap=int(swap),
                                         gain=final.loc["pm", "mean_return_pct"] - final.loc["control", "mean_return_pct"],
                                         pm_return=final.loc["pm", "mean_return_pct"],
                                         control_return=final.loc["control", "mean_return_pct"], **candidate))
                        pd.DataFrame(rows).to_csv(self.output / "parameter_search.csv", index=False)
                print(f"Validation {family}, candidate {index + 1}/{len(self.protocol['candidates'])} complete", flush=True)
            data = pd.DataFrame(rows)
            means = data[data.family == family].groupby("candidate", sort=True).gain.mean()
            chosen = int(means.idxmax())
            self.selection[family] = dict(candidate=chosen, **self.protocol["candidates"][chosen])
        self.tuning_results = pd.DataFrame(rows)
        self.manifest["selected_parameters"] = self.selection
        self._manifest()
        print("Selected on validation only:", self.selection, flush=True)

    def evaluate(self):
        if not self.selection or self.evaluated:
            raise RuntimeError("Tune first; run held-out evaluation once")
        self.evaluated = True
        snapshots, agents, fits = [], [], []
        for seed in self.protocol["test_seeds"]:
            markets = [("all_control", False, dict(weight=0.0, discount=0.0))] + [
                (family, swap, self.selection[family]) for family in ["process", "no_sequence"] for swap in [False, True]]
            for family, swap, candidate in markets:
                s, a, f = self.run_market(seed, family, candidate, swap, stage="test", record=True)
                snapshots.append(s); agents.append(a); fits.append(f)
                print(f"Test seed {seed}, {family}, swap={swap} complete", flush=True)
        self.results = pd.concat(snapshots, ignore_index=True)
        self.agents = pd.concat(agents, ignore_index=True)
        self.fit_audit = pd.concat([f for f in fits if not f.empty], ignore_index=True)
        # Complementary assignments are dependent replicates, not extra seeds.
        mixed = self.results[self.results.family != "all_control"]
        wide = mixed.pivot(index=["seed", "family", "swap", "day", "feedback_round"], columns="cohort", values=METRICS)
        differences = pd.DataFrame({metric: wide[metric]["pm"] - wide[metric]["control"] for metric in METRICS}).reset_index()
        self.seed_effects = differences.groupby(["seed", "family", "day", "feedback_round"], as_index=False)[METRICS].mean()
        rows = []
        for (family, day, round_number), group in self.seed_effects.groupby(["family", "day", "feedback_round"]):
            for metric in METRICS:
                rows.append(dict(family=family, day=day, feedback_round=round_number, metric=metric,
                                 **bootstrap(group[metric], self.protocol["bootstrap_samples"])))
        self.effects = pd.DataFrame(rows)
        final = self.results[self.results.day == self.days]
        base = final[final.family == "all_control"].set_index("seed")
        welfare = []
        for family in ["process", "no_sequence"]:
            mixed_final = final[final.family == family].groupby(["seed", "cohort"]).mean_return_pct.mean().unstack()
            for seed, row in mixed_final.iterrows():
                welfare.append(dict(seed=seed, family=family,
                                    pm_vs_all_control=row.pm - base.loc[seed, "mean_return_pct"],
                                    control_spillover=row.control - base.loc[seed, "mean_return_pct"],
                                    market_vs_all_control=(row.pm + row.control) / 2 - base.loc[seed, "mean_return_pct"]))
        self.welfare = pd.DataFrame(welfare)
        effects_final = self.seed_effects[self.seed_effects.day == self.days].pivot(index="seed", columns="family", values="mean_return_pct")
        self.sequence_effect = pd.DataFrame([bootstrap(effects_final.process - effects_final.no_sequence,
                                                       self.protocol["bootstrap_samples"])])
        self.export()

    def export(self):
        for name, data in {
            "learning_curve": self.effects, "seed_effects": self.seed_effects,
            "held_out_results": self.results, "agent_outcomes": self.agents,
            "fit_audit": self.fit_audit, "run_audit": pd.DataFrame(self.audit),
            "welfare_comparison": self.welfare, "sequence_effect": self.sequence_effect,
        }.items():
            data.to_csv(self.output / f"{name}.csv", index=False)
        self.manifest["completed"] = True
        self.manifest["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                         for name in ["online_feedback_experiment.py", "process_policy.py", "mining_hardware.py"]}
        notebook = json.loads(Path("notebook.ipynb").read_text(encoding="utf-8"))
        self.manifest["notebook_source_sha256"] = hashlib.sha256(json.dumps(
            [c["source"] for c in notebook["cells"]], ensure_ascii=False).encode()).hexdigest()
        self._manifest()

    def conclusion(self):
        row = self.effects[(self.effects.family == "process") & (self.effects.day == self.days) &
                           (self.effects.metric == "mean_return_pct")].iloc[0]
        result = ("The paired interval supports a positive average relative advantage." if row.ci_low > 0 else
                  "A positive average relative advantage is not established by this interval.")
        welfare = bootstrap(self.welfare[self.welfare.family == "process"].market_vs_all_control,
                            self.protocol["bootstrap_samples"])
        return (f"{'SMOKE TEST ONLY. ' if self.protocol['smoke'] else ''}After {self.protocol['feedback_updates']} feedback updates "
                f"(day {self.days}), PM minus concurrent controls: {row.mean_delta:+.2f} percentage points "
                f"(paired seed bootstrap 95% interval {row.ci_low:+.2f} to {row.ci_high:+.2f}); "
                f"positive on {int(row.positive_seeds)}/{int(row.paired_seeds)} seeds after averaging complementary assignments. "
                f"{result} Overall market return versus an all-control market: {welfare['mean_delta']:+.2f} points "
                f"(interval {welfare['ci_low']:+.2f} to {welfare['ci_high']:+.2f}). "
                "This concerns relative advantage at 50% adoption in the simulated scenario, not a guarantee for every agent or greater total welfare.")

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for family, label, color in [("process", "Sequence PM", "#0072B2"), ("no_sequence", "No-sequence ablation", "#D55E00")]:
            curve = self.effects[(self.effects.family == family) & (self.effects.metric == "mean_return_pct")].sort_values("feedback_round")
            axes[0].plot(curve.feedback_round, curve.mean_delta, marker="o", label=label, color=color)
            axes[0].fill_between(curve.feedback_round, curve.ci_low, curve.ci_high, alpha=.16, color=color)
        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].set(xlabel="Completed feedback updates", ylabel="PM minus controls (percentage points)", title="Online feedback learning curve (95% intervals)")
        axes[0].legend(fontsize=8)
        for cohort, label, color in [("pm", "PM agents", "#0072B2"), ("control", "Concurrent controls", "#777777")]:
            data = self.results[(self.results.family == "process") & (self.results.cohort == cohort)]
            curve = data.groupby("day").mean_return_pct.mean()
            axes[1].plot(curve.index, curve.values, label=label, color=color)
        base = self.results[self.results.family == "all_control"].groupby("day").mean_return_pct.mean()
        axes[1].plot(base.index, base.values, "--", color="black", label="All-control market")
        axes[1].set(xlabel="Simulated day", ylabel="Mean economic return (%)", title="Same agents continue across feedback rounds")
        axes[1].legend(fontsize=8)
        last = self.seed_effects[(self.seed_effects.family == "process") & (self.seed_effects.day == self.days)]
        axes[2].bar(last.seed.astype(str), last.mean_return_pct, color=np.where(last.mean_return_pct > 0, "#0072B2", "#D55E00"))
        axes[2].axhline(0, color="gray", linestyle="--")
        axes[2].set(xlabel="Held-out scenario seed", ylabel="PM minus controls (percentage points)", title="All test seeds, complementary assignments averaged")
        axes[2].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(self.output / "online_learning_curve.png", dpi=180, bbox_inches="tight")
        fig.savefig(self.output / "online_learning_curve.pdf", bbox_inches="tight")
        plt.show()
