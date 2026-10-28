"""Reproducible validation search and iterative PM experiment for notebook section 13.

Model definitions remain in the notebook. A checkpoint transfers mined advice to
fresh populations; individual agent values are not carried between simulations.
"""
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

from process_feedback import ProcessAdviser, paired_effects


METRICS = ["mean_return_pct", "median_return_pct", "mean_max_drawdown_pct",
           "bankruptcy_pct", "survival_pct"]
LOG_COLUMNS = [
    "run_id", "agent_id", "cycle", "context", "previous_action", "action",
    "normalized_reward", "terminal", "capital_context", "capital_action",
    "previous_capital_action", "capital_wealth_before", "net_worth_before",
    "initial_net_worth",
]


def default_protocol(smoke=False):
    """Declare the entire search and seed budget before inspecting test outcomes."""
    rounds = 2 if smoke else 6
    return {
        "smoke": smoke, "n_agents": 6 if smoke else 40,
        "training_days": 12 if smoke else 365, "rounds": rounds,
        "training_seeds": [[4100 + 100 * r + i for i in range(1, 3 if smoke else 4)]
                           for r in range(rounds)],
        "validation_seeds": [5101, 5102] if smoke else [5101, 5102, 5103],
        "evaluation_seeds": [6101, 6102] if smoke else list(range(6101, 6109)),
        "evaluation_horizons": [6, 12] if smoke else [365, 1095],
        "primary_horizon": 12 if smoke else 365,
        "primary_metric": "mean_return_pct",
        "core_candidates": [dict(learning_rate=lr, exploration_rate=eps)
                            for lr, eps in product([0.08, 0.18], [0.09] if smoke else [0.04, 0.09])],
        # Weight zero is the control reference, not an active PM treatment.
        "advice_candidates": [{"pm_weight": 0.0, "shrinkage": 5.0,
                               "operating_horizon": 3 if smoke else 30}] + [
            dict(pm_weight=w, shrinkage=s, operating_horizon=h)
            for h, w, s in product([3] if smoke else [1, 7, 30],
                                   [1.0] if smoke else [1.0, 3.0, 6.0],
                                   [5.0] if smoke else [5.0, 25.0])],
        "operating_horizon": 3 if smoke else 30,
        "capital_horizon": 4 if smoke else 180,
        "min_support": 2 if smoke else 10, "capital_min_support": 2 if smoke else 5,
        "min_agents": 2 if smoke else 3, "min_runs": 2,
        "bootstrap_samples": 200 if smoke else 5000,
    }


def validate_protocol(p):
    groups = p["training_seeds"] + [p["validation_seeds"], p["evaluation_seeds"]]
    flattened = [seed for group in groups for seed in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Training rounds, validation and evaluation must use disjoint seeds")
    if p["rounds"] != len(p["training_seeds"]) or p["rounds"] < 2:
        raise ValueError("At least two feedback rounds with explicit seeds are required")
    if any(len(group) < 2 for group in groups):
        raise ValueError("Each stage needs at least two independent seeds")
    if p["primary_horizon"] not in p["evaluation_horizons"]:
        raise ValueError("Primary horizon must be evaluated")
    if p["training_days"] < max(p["operating_horizon"], p["capital_horizon"]):
        raise ValueError("Training must cover complete adviser outcome windows")


def feedback_metrics(model, seed, variant, horizon_days):
    agents = model.agent_summary_df()
    decisions = model.decision_log_df()
    reviews = decisions[decisions.capital_action != "no_review"]
    return {
        "seed": seed, "variant": variant, "horizon_days": horizon_days,
        "mean_return_pct": 100 * agents.normalized_return.mean(),
        "median_return_pct": 100 * agents.normalized_return.median(),
        "mean_max_drawdown_pct": 100 * agents.max_drawdown.mean(),
        "bankruptcy_pct": 100 * agents.exit_reason.eq("bankruptcy").mean(),
        "survival_pct": 100 * agents.status.eq("active").mean(),
        "advice_coverage_pct": 100 * decisions.pm_support.gt(0).mean(),
        "capital_advice_coverage_pct": 100 * reviews.capital_pm_support.gt(0).mean() if len(reviews) else 0.0,
        "mean_capital_updates": agents.capital_learning_updates.mean(),
    }


def select_candidate(rows):
    """Maximize mean seed-level return; deterministic ties use candidate order."""
    summary = rows.groupby("candidate", sort=True).mean(numeric_only=True)
    if not np.isfinite(summary.mean_return_pct).all():
        raise ValueError("Non-finite validation objective")
    return int(summary.mean_return_pct.idxmax())


def model_fingerprint():
    notebook = json.loads(Path("notebook.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    definitions = [s for s in sources if s.startswith((
        "from __future__ import annotations", "def gini(", "@dataclass\nclass MiningParams",
        "class AdaptiveMiningAgent(", "class AdaptiveMiningModel("))]
    text = "\n".join(definitions) + Path("process_feedback.py").read_text() + Path("mining_hardware.py").read_text()
    return hashlib.sha256(text.encode()).hexdigest()


class FeedbackExperiment:
    def __init__(self, model_class, base_params, protocol, output):
        validate_protocol(protocol)
        self.model_class = model_class
        self.protocol = json.loads(json.dumps(protocol))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        # Keep the pre-existing stress scenario. Scale issuance with population
        # to preserve expected initial reward per miner when reducing N=150.
        self.params = replace(
            base_params, n_agents=protocol["n_agents"], cycles=protocol["training_days"],
            daily_reward_coins=base_params.daily_reward_coins * protocol["n_agents"] / base_params.n_agents,
            cost_per_hash_factor=6.2, energy_shock_prob=0.14,
            energy_shock_multiplier_min=2.5, energy_shock_multiplier_max=4.5,
            pm_weight=0.0,
        )
        self.checkpoints = {"process": {}, "no_sequence": {}}
        self.corpora = {}
        self.selection = {}
        self.training_rows = []
        self.tuning_rows = []
        self.audit_rows = []
        self._evaluated = False
        self.manifest = {
            "protocol": self.protocol, "scenario_before_tuning": asdict(self.params),
            "selection_rule": "Maximum mean validation return within each active treatment; zero advice is the control. First candidate breaks ties.",
            "model_fingerprint": model_fingerprint(),
            "primary_comparison": "Final process checkpoint minus tuned baseline at primary_horizon",
            "test_policy": "All declared checkpoints reported; no test-based selection or stopping",
            "development_note": "Initial validation search at operating horizon 30 selected zero advice. "
                                "The grid was expanded to horizons 1/7/30 using validation evidence only, "
                                "before inspecting new test outcomes. This is not preregistration.",
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ["mesa", "numpy", "pandas", "matplotlib"]},
        }
        self._write_manifest()

    def _write_manifest(self):
        (self.output / "experiment_config.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _model(self, seed, stage, advisers=(None, None), weight=0.0, **overrides):
        self.audit_rows.append({"stage": stage, "seed": seed, "pm_weight": weight})
        return self.model_class(
            replace(self.params, seed=seed, pm_weight=weight, **overrides),
            process_adviser=advisers[0], capital_adviser=advisers[1], run_id=f"{stage}-{seed}",
        )

    def _fit(self, data, arm, candidate):
        p = self.protocol
        common = dict(min_agents=p["min_agents"], min_runs=p["min_runs"],
                      shrinkage=candidate["shrinkage"], use_sequence=arm == "process")
        return (
            ProcessAdviser(horizon=candidate["operating_horizon"], min_support=p["min_support"], **common).fit(data),
            ProcessAdviser(horizon=p["capital_horizon"], min_support=p["capital_min_support"],
                           decision_kind="capital", **common).fit(data),
        )

    def _training_batch(self, round_number, arm, advisers=(None, None), weight=0.0):
        logs = []
        for seed in self.protocol["training_seeds"][round_number - 1]:
            model = self._model(seed, f"train-{arm}-round{round_number}", advisers, weight)
            model.run_model()
            data = model.decision_log_df()[LOG_COLUMNS]
            logs.append(data)
            self.training_rows.append({
                **feedback_metrics(model, seed, arm, self.params.cycles),
                "round": round_number, "agent_days": len(data),
            })
            print(f"Training {arm}, round {round_number}, seed {seed}: {len(data):,} agent-days", flush=True)
        return pd.concat(logs, ignore_index=True)

    def tune(self, validation_cache=None):
        if self.selection:
            raise RuntimeError("Create a fresh experiment to tune again")
        p = self.protocol
        cached = None
        if validation_cache is not None:
            path = Path(validation_cache)
            config = json.loads((path.parent / "experiment_config.json").read_text(encoding="utf-8"))
            for key in ["protocol", "scenario_before_tuning", "packages", "model_fingerprint"]:
                if config.get(key) != self.manifest[key]:
                    raise ValueError(f"Validation cache does not match {key}")
            cached = pd.read_csv(path)
            expected = {(search, i, seed) for search, candidates in [
                ("core", p["core_candidates"]), ("process", p["advice_candidates"]),
                ("no_sequence", p["advice_candidates"])]
                for i in range(len(candidates)) for seed in p["validation_seeds"]}
            actual = set(cached[["search", "candidate", "seed"]].itertuples(index=False, name=None))
            if actual != expected or len(cached) != len(expected) or not np.isfinite(cached.mean_return_pct).all():
                raise ValueError("Validation cache must contain every candidate/seed exactly once")
            self.manifest["validation_cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        # Give the unguided baseline the benefit of selecting its own learner.
        # PM and ablation inherit exactly this same learner configuration.
        for index, candidate in enumerate(p["core_candidates"]):
            for seed in p["validation_seeds"]:
                if cached is not None:
                    self.tuning_rows.append(cached[(cached.search == "core") & (cached.candidate == index) & (cached.seed == seed)].iloc[0].to_dict())
                    self.audit_rows.append({"stage": "validation-core-cached", "seed": seed, "pm_weight": 0.0})
                    continue
                model = self._model(seed, "validation-core", **candidate)
                model.run_model()
                self.tuning_rows.append({
                    **feedback_metrics(model, seed, "baseline", self.params.cycles),
                    "search": "core", "candidate": index, **candidate,
                })
            print(f"Core parameter candidate {index + 1}/{len(p['core_candidates'])} evaluated", flush=True)
        core_index = select_candidate(pd.DataFrame(self.tuning_rows))
        self.params = replace(self.params, **p["core_candidates"][core_index])
        self.selection["core"] = {"candidate": core_index, **p["core_candidates"][core_index]}
        initial_data = self._training_batch(1, "shared")
        for arm in self.checkpoints:
            for index, candidate in enumerate(p["advice_candidates"]):
                advisers = self._fit(initial_data, arm, candidate)
                for seed in p["validation_seeds"]:
                    if cached is not None:
                        self.tuning_rows.append(cached[(cached.search == arm) & (cached.candidate == index) & (cached.seed == seed)].iloc[0].to_dict())
                        self.audit_rows.append({"stage": f"validation-{arm}-cached", "seed": seed, "pm_weight": candidate["pm_weight"]})
                        continue
                    model = self._model(seed, f"validation-{arm}", advisers, candidate["pm_weight"])
                    model.run_model()
                    self.tuning_rows.append({
                        **feedback_metrics(model, seed, arm, self.params.cycles),
                        "search": arm, "candidate": index, **candidate,
                    })
                print(f"{arm} candidate {index + 1}/{len(p['advice_candidates'])} evaluated", flush=True)
            rows = pd.DataFrame(self.tuning_rows)
            # Estimate the effect of an active PM intervention even if validation
            # prefers the control. The negative validation evidence stays visible.
            chosen = select_candidate(rows[(rows.search == arm) & (rows.pm_weight > 0)])
            self.selection[arm] = {"candidate": chosen, **p["advice_candidates"][chosen]}
            self.corpora[arm] = initial_data.copy()
            self.checkpoints[arm][1] = self._fit(initial_data, arm, self.selection[arm])
        self.tuning_results = pd.DataFrame(self.tuning_rows)
        self.tuning_results.to_csv(self.output / "parameter_search.csv", index=False)
        self.manifest.update(selected_parameters=self.selection, selected_model_params=asdict(self.params))
        self._write_manifest()
        print("Validation selection (frozen before later rounds and test evaluation):", self.selection, flush=True)

    def train(self):
        if not self.selection or any(len(c) != 1 for c in self.checkpoints.values()):
            raise RuntimeError("Tune once before training the remaining rounds")
        for round_number in range(2, self.protocol["rounds"] + 1):
            for arm in self.checkpoints:
                data = self._training_batch(round_number, arm, self.checkpoints[arm][round_number - 1],
                                            self.selection[arm]["pm_weight"])
                self.corpora[arm] = pd.concat([self.corpora[arm], data], ignore_index=True)
                self.checkpoints[arm][round_number] = self._fit(self.corpora[arm], arm, self.selection[arm])
        self.training_results = pd.DataFrame(self.training_rows)
        self.training_results.to_csv(self.output / "training_phase_stats.csv", index=False)
        for arm, checkpoints in self.checkpoints.items():
            # These are the actual fitting records, including censored windows;
            # ProcessAdviser excludes incomplete outcome windows itself.
            self.corpora[arm].to_json(self.output / f"{arm}_training_decisions.jsonl.gz",
                                     orient="records", lines=True, compression="gzip")
            for round_number, advisers in checkpoints.items():
                for kind, adviser in zip(["operating", "capital"], advisers):
                    adviser.table.to_csv(self.output / f"{arm}_{kind}_edges_round{round_number}.csv", index=False)

    def evaluate(self):
        if self._evaluated or any(len(c) != self.protocol["rounds"] for c in self.checkpoints.values()):
            raise RuntimeError("Finish all training rounds before a single held-out evaluation")
        self._evaluated = True
        variants = [("baseline", 0, (None, None), 0.0)]
        for round_number, advisers in self.checkpoints["process"].items():
            variants.append((f"pm_round_{round_number}", round_number, advisers, self.selection["process"]["pm_weight"]))
        final_round = self.protocol["rounds"]
        variants.append(("no_sequence", final_round, self.checkpoints["no_sequence"][final_round],
                         self.selection["no_sequence"]["pm_weight"]))
        rows, trajectories, agents_rows = [], [], []
        for seed in self.protocol["evaluation_seeds"]:
            for variant, round_number, advisers, weight in variants:
                model = self._model(seed, f"test-{variant}", advisers, weight,
                                    cycles=max(self.protocol["evaluation_horizons"]))
                for day in range(1, model.params.cycles + 1):
                    model.step()
                    if day in self.protocol["evaluation_horizons"]:
                        rows.append({**feedback_metrics(model, seed, variant, day), "round": round_number})
                        agents_rows.append(model.agent_summary_df().assign(seed=seed, variant=variant, horizon_days=day))
                model.finalize_run()
                trajectory = model.decision_log_df().groupby("cycle").normalized_reward.sum().cumsum()
                # Exited agents have frozen wealth; retain the full calendar.
                trajectory = trajectory.reindex(range(1, model.params.cycles + 1)).ffill().fillna(0)
                trajectories.extend({"seed": seed, "variant": variant, "day": day,
                                     "mean_return_pct": 100 * value / self.params.n_agents}
                                    for day, value in trajectory.items())
                print(f"Test {variant}, seed {seed} complete", flush=True)
        self.results = pd.DataFrame(rows)
        self.trajectories = pd.DataFrame(trajectories)
        self.agent_results = pd.concat(agents_rows, ignore_index=True)
        bootstrap = self.protocol["bootstrap_samples"]
        self.effects = pd.concat([
            paired_effects(group, METRICS, bootstrap_samples=bootstrap).assign(horizon_days=days)
            for days, group in self.results.groupby("horizon_days")
        ], ignore_index=True)
        comparisons = []
        for days, group in self.results.groupby("horizon_days"):
            for reference in ["no_sequence", "pm_round_1"]:
                subset = group[group.variant.isin([f"pm_round_{final_round}", reference])]
                comparisons.append(paired_effects(subset, METRICS, bootstrap_samples=bootstrap,
                                                  reference=reference).assign(horizon_days=days))
        self.comparisons = pd.concat(comparisons, ignore_index=True)
        self.export()

    def export(self):
        for name, data in {
            "held_out_results": self.results, "paired_effects": self.effects,
            "secondary_comparisons": self.comparisons, "return_trajectories": self.trajectories,
            "agent_outcomes": self.agent_results, "run_audit": pd.DataFrame(self.audit_rows),
        }.items():
            data.to_csv(self.output / f"{name}.csv", index=False)
        self.manifest["completed"] = True
        self.manifest["source_sha256"] = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ["feedback_experiment.py", "process_feedback.py", "mining_hardware.py"]
        }
        # Hash notebook sources, not saved outputs, so embedding results is safe.
        notebook = json.loads(Path("notebook.ipynb").read_text(encoding="utf-8"))
        self.manifest["notebook_source_sha256"] = hashlib.sha256(
            json.dumps([c["source"] for c in notebook["cells"]], ensure_ascii=False).encode()
        ).hexdigest()
        self._write_manifest()

    def conclusion(self):
        p = self.protocol
        row = self.effects[
            (self.effects.variant == f"pm_round_{p['rounds']}") &
            (self.effects.metric == p["primary_metric"]) &
            (self.effects.horizon_days == p["primary_horizon"])
        ].iloc[0]
        primary = self.results[self.results.horizon_days == p["primary_horizon"]]
        wide = primary.pivot(index="seed", columns="variant", values=p["primary_metric"])
        wins = int((wide[f"pm_round_{p['rounds']}"] > wide.baseline).sum())
        if row.ci_low > 0:
            interpretation = "The paired interval supports a positive average effect in this scenario."
        elif row.ci_high < 0:
            interpretation = "The paired interval supports a negative average effect in this scenario."
        else:
            interpretation = "The paired interval includes zero; an average improvement is not established."
        return (
            f"{'SMOKE TEST ONLY. ' if p['smoke'] else ''}Primary endpoint: final PM minus tuned baseline "
            f"at day {p['primary_horizon']}: {row.mean_delta:+.2f} percentage points "
            f"(paired seed bootstrap 95% interval {row.ci_low:+.2f} to {row.ci_high:+.2f}). "
            f"PM has a higher mean return on {wins}/{len(wide)} test seeds. {interpretation} "
            "Intervals condition on this trained adviser and are not a guarantee for every agent or seed. "
            "Intermediate rounds, other metrics and the longer horizon are exploratory; no multiplicity correction is applied."
        )

    def plot(self):
        import matplotlib.pyplot as plt

        p = self.protocol
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for days, color in zip(p["evaluation_horizons"], ["#0072B2", "#D55E00"]):
            data = self.effects[(self.effects.horizon_days == days) &
                                (self.effects.metric == p["primary_metric"]) &
                                self.effects.variant.str.startswith("pm_round_")].copy()
            data["round"] = data.variant.str.rsplit("_", n=1).str[-1].astype(int)
            data = data.sort_values("round")
            axes[0].errorbar(data["round"], data.mean_delta,
                            yerr=[data.mean_delta - data.ci_low, data.ci_high - data.mean_delta],
                            marker="o", capsize=3, label=f"Day {days}", color=color)
        axes[0].axhline(0, color="gray", linestyle="--")
        axes[0].set(xlabel="Completed PM fitting rounds", ylabel="Mean return difference (percentage points)",
                    title="Feedback learning curve vs tuned baseline")
        axes[0].set_xticks(range(1, p["rounds"] + 1))
        axes[0].legend()
        selected = ["baseline", "pm_round_1", f"pm_round_{p['rounds']}", "no_sequence"]
        for variant in selected:
            data = self.trajectories[self.trajectories.variant == variant]
            daily = data.groupby("day").mean_return_pct.mean()
            axes[1].plot(daily.index, daily.values, label=variant)
        axes[1].set(xlabel="Simulated day", ylabel="Mean economic return (%)",
                    title="Within-run return trajectories")
        axes[1].legend(fontsize=8)
        data = self.results[self.results.horizon_days == p["primary_horizon"]]
        wide = data.pivot(index="seed", columns="variant", values=p["primary_metric"])
        for _, seed_row in wide.iterrows():
            axes[2].plot(range(3), seed_row[["baseline", f"pm_round_{p['rounds']}", "no_sequence"]],
                         color="gray", alpha=0.6, marker="o")
        axes[2].set_xticks(range(3), ["Tuned baseline", "Final PM", "No sequence"], rotation=15)
        axes[2].set(ylabel="Mean economic return (%)", title=f"Paired scenarios at day {p['primary_horizon']}")
        fig.tight_layout()
        fig.savefig(self.output / "feedback_learning_curve.png", dpi=180, bbox_inches="tight")
        fig.savefig(self.output / "feedback_learning_curve.pdf", bbox_inches="tight")
        plt.show()
        return fig
