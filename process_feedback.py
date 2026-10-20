"""Outcome-annotated decision transitions used by the mining notebook.

The adviser is an observational, bounded policy bias, not a causal estimator.
It is fitted on completed training runs and never updated during evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decision_outcomes(decisions: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """Attach forward returns without crossing agents/runs or incomplete windows.

    A terminal exit has zero subsequent return (assets are frozen). A simulation
    time limit is censoring: incomplete windows at that limit are excluded.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    required = {
        "run_id", "agent_id", "cycle", "context", "previous_action", "action",
        "normalized_reward", "terminal",
    }
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"Missing decision columns: {sorted(missing)}")
    if decisions.duplicated(["run_id", "agent_id", "cycle"]).any():
        raise ValueError("Each agent may have only one decision per cycle")

    parts = []
    for _, group in decisions.groupby(["run_id", "agent_id"], sort=False):
        group = group.sort_values("cycle").copy()
        if len(group) > 1 and not np.all(np.diff(group["cycle"]) == 1):
            raise ValueError("Decision histories must contain consecutive cycles")
        if group["terminal"].iloc[:-1].any():
            raise ValueError("A terminal decision must end the agent history")
        rewards = group["normalized_reward"].to_numpy(dtype=float)
        if not np.isfinite(rewards).all():
            raise ValueError("Rewards must be finite")
        prefix = np.concatenate(([0.0], np.cumsum(rewards)))
        start = np.arange(len(group))
        stop = np.minimum(start + horizon, len(group))
        group["forward_return"] = prefix[stop] - prefix[start]
        group["observed_cycles"] = stop - start
        complete = (stop - start == horizon) | bool(group["terminal"].iloc[-1])
        parts.append(group.loc[complete])
    if not parts:
        return decisions.assign(
            forward_return=pd.Series(dtype=float),
            observed_cycles=pd.Series(dtype=int),
        )
    return pd.concat(parts, ignore_index=True)


class ProcessAdviser:
    """Mine context-specific previous-action -> action edges and their outcomes.

    use_sequence=False is the state/outcome-only ablation: it uses exactly the
    same records, horizon and scoring rules, but removes previous-action history.
    """

    def __init__(
        self, horizon=30, min_support=20, min_agents=5, min_runs=2,
        shrinkage=50.0, use_sequence=True, decision_kind="operating",
    ):
        if min(horizon, min_support, min_agents, min_runs) < 1 or shrinkage <= 0:
            raise ValueError("Support thresholds and shrinkage must be positive")
        self.horizon = horizon
        self.min_support = min_support
        self.min_agents = min_agents
        self.min_runs = min_runs
        self.shrinkage = shrinkage
        self.use_sequence = use_sequence
        if decision_kind not in {"operating", "capital"}:
            raise ValueError("Unknown decision kind")
        self.decision_kind = decision_kind
        self.table = pd.DataFrame()
        self.training_run_ids = frozenset()
        self._lookup = {}
        self.return_scale = 1.0

    def fit(self, decisions: pd.DataFrame):
        data = decision_outcomes(decisions, self.horizon)
        if self.decision_kind == "capital":
            # Compute financial windows on ALL days before selecting review days.
            # Review histories span calendar gaps; the economic log must not.
            data = data[data["capital_action"] != "no_review"].copy()
            data["context"] = data["capital_context"]
            data["previous_action"] = data["previous_capital_action"]
            data["action"] = data["capital_action"]
            data["forward_return"] -= (
                data["capital_wealth_before"] - data["net_worth_before"]
            ) / data["initial_net_worth"]
        self.training_run_ids = frozenset(decisions["run_id"].unique())
        self._lookup = {}
        self.table = pd.DataFrame()
        if data.empty:
            return self
        # JSONL restores tuple contexts as lists.
        data["context"] = data["context"].map(tuple)
        data["history"] = data["previous_action"] if self.use_sequence else "*"
        data["agent_case"] = list(zip(data["run_id"], data["agent_id"]))
        keys = ["context", "history"]
        table = data.groupby(keys + ["action"], sort=False).agg(
            support=("forward_return", "size"),
            agents=("agent_case", "nunique"),
            runs=("run_id", "nunique"),
            mean_return=("forward_return", "mean"),
        ).reset_index()
        baseline = data.groupby(keys, sort=False)["forward_return"].mean().rename("context_return")
        table = table.join(baseline, on=keys)
        table["supported"] = (
            (table["support"] >= self.min_support)
            & (table["agents"] >= self.min_agents)
            & (table["runs"] >= self.min_runs)
        )
        # Avoid recommending from a context with only one supported choice.
        alternatives = table.groupby(keys, sort=False)["supported"].transform("sum")
        table["supported"] &= alternatives >= 2
        scale = float(data["forward_return"].std(ddof=0))
        self.return_scale = max(scale, 1e-6)
        table["confidence"] = np.where(
            table["supported"],
            table["support"] / (table["support"] + self.shrinkage),
            0.0,
        )
        table["score"] = table["confidence"] * np.tanh(
            (table["mean_return"] - table["context_return"]) / self.return_scale
        )
        self.table = table
        self._lookup = {
            (row.context, row.history, row.action): (float(row.score), int(row.support))
            for row in table.itertuples() if row.supported
        }
        return self

    def lookup(self, context, previous_action, action):
        """Return a bounded score and support; unknown edges have no influence."""
        history = previous_action if self.use_sequence else "*"
        return self._lookup.get((context, history, action), (0.0, 0))


def paired_effects(results: pd.DataFrame, metrics, bootstrap_samples=2000, reference="baseline"):
    """Seed-level paired differences and descriptive bootstrap 95% intervals.

    Seeds, not agents, are the independent experimental units. Small seed sets
    give unstable intervals; these are pilot estimates, not significance tests.
    """
    if results.duplicated(["seed", "variant"]).any():
        raise ValueError("Expected one result per seed and variant")
    if reference not in set(results["variant"]):
        raise ValueError(f"Missing reference variant: {reference}")
    rng = np.random.default_rng(1701)
    rows = []
    for variant in results["variant"].unique():
        if variant == reference:
            continue
        for metric in metrics:
            wide = results.pivot(index="seed", columns="variant", values=metric)
            pair = wide[[variant, reference]]
            if pair.isna().any().any():
                raise ValueError("All variants must have complete metrics on the same seeds")
            delta = (pair[variant] - pair[reference]).to_numpy()
            if len(delta) < 2:
                raise ValueError("At least two paired evaluation seeds are required")
            boot = rng.choice(delta, size=(bootstrap_samples, len(delta)), replace=True).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "variant": variant, "reference": reference,
                "metric": metric, "paired_seeds": len(delta),
                "mean_delta": delta.mean(), "ci_low": low, "ci_high": high,
            })
    return pd.DataFrame(rows)
