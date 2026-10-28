"""Learn a policy from an outcome-annotated, discovered transition system.

Action labels are opaque: this module contains no mining-domain decision rules.
Only observed decision events, their ordering and their realized rewards enter
discovery. Planning is over the empirical graph, not the simulator's internals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class TransitionProcessAdviser:
    def __init__(self, discount=0.9, min_support=5, min_agents=2,
                 shrinkage=10.0, use_sequence=True, reward_column="learning_reward"):
        if not 0 <= discount < 1 or min_support < 1 or min_agents < 1 or shrinkage <= 0:
            raise ValueError("Invalid transition-policy parameters")
        self.discount = discount
        self.min_support = min_support
        self.min_agents = min_agents
        self.shrinkage = shrinkage
        self.use_sequence = use_sequence
        self.reward_column = reward_column
        self.table = pd.DataFrame()
        self.transition_table = pd.DataFrame()
        self._lookup = {}
        self.last_observed_cycle = 0

    def fit(self, decisions):
        required = {"run_id", "agent_id", "cycle", "context", "previous_action",
                    "action", "terminal", self.reward_column}
        if not required.issubset(decisions.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(decisions.columns))}")
        if decisions.duplicated(["run_id", "agent_id", "cycle"]).any():
            raise ValueError("Duplicate decision events")
        self._lookup = {}
        self.table = pd.DataFrame()
        self.transition_table = pd.DataFrame()
        if decisions.empty:
            return self
        data = decisions.sort_values(["run_id", "agent_id", "cycle"]).copy()
        if not np.isfinite(data[self.reward_column]).all():
            raise ValueError("Nonfinite reward")
        data["context"] = data.context.map(tuple)
        data["history"] = data.previous_action if self.use_sequence else "*"
        data["state"] = list(zip(data.context, data.history))
        groups = data.groupby(["run_id", "agent_id"], sort=False)
        next_cycle = groups.cycle.shift(-1)
        if ((next_cycle.notna()) & (next_cycle != data.cycle + 1)).any():
            raise ValueError("Decision histories must be consecutive")
        if (data.terminal & next_cycle.notna()).any():
            raise ValueError("Terminal event must end its case")
        data["next_state"] = groups.state.shift(-1)
        # A boundary without a next event is censored, not a terminal transition.
        data = data[data.terminal | next_cycle.notna()].copy()
        if data.empty:
            return self
        self.last_observed_cycle = int(decisions.cycle.max())
        data["case"] = list(zip(data.run_id, data.agent_id))
        terminal_state = ("__terminal__",)
        data.loc[data.terminal, "next_state"] = pd.Series(
            [terminal_state] * int(data.terminal.sum()), index=data.index[data.terminal], dtype=object)
        keys = ["state", "action"]
        table = data.groupby(keys, sort=False).agg(
            support=(self.reward_column, "size"), agents=("case", "nunique"),
            mean_reward=(self.reward_column, "mean"),
        ).reset_index()
        table["supported"] = (table.support >= self.min_support) & (table.agents >= self.min_agents)
        transitions = data.groupby(keys + ["next_state"], sort=False).size().rename("frequency").reset_index()
        table["edge_id"] = np.arange(len(table))
        transitions = transitions.merge(table[keys + ["edge_id", "support"]], on=keys, validate="many_to_one")
        transitions["probability"] = transitions.frequency / transitions.support
        states = list(dict.fromkeys(table.state.tolist() + transitions.next_state.tolist()))
        state_index = {s: i for i, s in enumerate(states)}
        from_state = table.state.map(lambda state: state_index[state]).to_numpy()
        transition_matrix = csr_matrix((transitions.probability,
                                       (transitions.edge_id, transitions.next_state.map(lambda state: state_index[state]))),
                                      shape=(len(table), len(states)))
        reward = table.mean_reward.to_numpy()
        supported = table.supported.to_numpy()
        value = np.zeros(len(states))
        for _ in range(1000):
            q = reward + self.discount * (transition_matrix @ value)
            new_value = np.full(len(states), -np.inf)
            np.maximum.at(new_value, from_state[supported], q[supported])
            new_value[~np.isfinite(new_value)] = 0.0
            if np.max(np.abs(new_value - value)) < 1e-10:
                value = new_value
                break
            value = new_value
        table["planned_return"] = reward + self.discount * (transition_matrix @ value)
        table["confidence"] = table.support / (table.support + self.shrinkage)
        table["score"] = 0.0
        # Compare supported alternatives locally. No global scale dominated by
        # rich survivors, and no hand-coded preference for any activity label.
        for _, rows in table[table.supported].groupby("state", sort=False):
            if len(rows) < 2:
                continue
            values = rows.planned_return
            spread = float(values.max() - values.min())
            if spread > 1e-12:
                centered = 2 * (values - values.min()) / spread - 1
                table.loc[rows.index, "score"] = centered * rows.confidence
        table["context"] = table.state.map(lambda s: s[0])
        table["history"] = table.state.map(lambda s: s[1])
        self._lookup = {(r.context, r.history, r.action): (float(r.score), int(r.support))
                        for r in table.itertuples() if r.supported}
        self.table = table
        self.transition_table = transitions
        return self

    def lookup(self, context, previous_action, action):
        return self._lookup.get((tuple(context), previous_action if self.use_sequence else "*", action), (0.0, 0))
