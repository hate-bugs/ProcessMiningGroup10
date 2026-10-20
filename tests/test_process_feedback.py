"""Financial invariants, adviser isolation and paired-scenario checks.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -v
The model definitions are executed directly from the notebook; no duplicate model.
"""
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

from process_feedback import ProcessAdviser, decision_outcomes, paired_effects


def load_notebook_model():
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebook.ipynb").read_text(encoding="utf-8"))
    module = types.ModuleType("mining_notebook_test")
    sys.modules[module.__name__] = module
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if text.startswith((
            "from __future__ import annotations", "def gini(",
            "@dataclass\nclass MiningParams", "class AdaptiveMiningAgent(",
            "class AdaptiveMiningModel(",
        )):
            exec(compile(text, "<notebook definitions>", "exec"), module.__dict__)
    return module


NB = load_notebook_model()


def model(**overrides):
    kwargs = dict(
        n_agents=1, cycles=1, seed=9, hardware_failure_prob=0,
        cost_per_hash_factor=0, block_reward_coins=0, imitation_rate=0,
    )
    kwargs.update(overrides)
    return NB.AdaptiveMiningModel(NB.MiningParams(**kwargs))


def force_action(m, action):
    for agent in m.agents:
        agent.choose_action = lambda: action


class AccountingTests(unittest.TestCase):
    def test_pause_without_miners_settles_once_without_pseudo_reward(self):
        m = model()
        agent = next(iter(m.agents))
        old = agent.action_values["pause"]
        force_action(m, "pause")
        m.run_model()
        self.assertEqual(len(m.decision_log), 1)
        self.assertEqual(m.decision_log[0]["economic_profit"], 0)
        self.assertAlmostEqual(agent.action_values["pause"], old * (1 - m.params.learning_rate))
        self.assertEqual(agent.recent_profits, [0])
        self.assertFalse(m.decision_log[0]["terminal"])  # time limit is censoring

    def test_investment_records_cash_spend_and_resale_haircut(self):
        m = model()
        agent = next(iter(m.agents))
        before_cash = agent.capital
        spend = min(before_cash * (0.04 + 0.24 * agent.risk_tolerance), before_cash - 40)
        expected_loss = spend * (m.params.resale_value_fraction * agent.hardware_efficiency - 1)
        force_action(m, "invest")
        m.run_model()
        self.assertAlmostEqual(agent.capital, before_cash - spend)
        self.assertAlmostEqual(m.decision_log[0]["economic_profit"], expected_loss)
        self.assertAlmostEqual(
            m.decision_log[0]["normalized_reward"], expected_loss / agent.initial_net_worth
        )

    def test_repair_is_asset_recovery_not_free_profit(self):
        m = model()
        agent = next(iter(m.agents))
        agent.damaged_hash = 2
        agent.needs_repair = True
        before_cash = agent.capital
        force_action(m, "repair")
        m.run_model()
        self.assertAlmostEqual(agent.capital, before_cash - 6)
        self.assertAlmostEqual(agent.damaged_hash, 0)
        self.assertAlmostEqual(m.decision_log[0]["economic_profit"], 0)

    def test_investment_reward_does_not_redefine_operating_loss_tolerance(self):
        m = model(cycles=15)
        force_action(m, "invest")
        m.run_model()
        agent = next(iter(m.agents))
        self.assertTrue(agent.active)
        self.assertEqual(agent.recent_profits, [0] * m.params.max_recent_profit_memory)
        self.assertLess(m.decision_log_df()["economic_profit"].sum(), 0)

    def test_sale_exchanges_assets_for_cash(self):
        m = model()
        agent = next(iter(m.agents))
        before_cash = agent.capital
        force_action(m, "sell_hardware")
        m.run_model()
        self.assertGreater(agent.capital, before_cash)
        self.assertAlmostEqual(m.decision_log[0]["economic_profit"], 0)

    def test_minimum_hash_cannot_create_sale_or_damage_value(self):
        m = model(hardware_failure_prob=1)
        agent = next(iter(m.agents))
        agent.hash_power = m.params.min_hash_power
        before_cash = agent.capital
        force_action(m, "sell_hardware")
        m.run_model()
        self.assertEqual(agent.hash_power, m.params.min_hash_power)
        self.assertEqual(agent.damaged_hash, 0)
        self.assertFalse(agent.needs_repair)
        self.assertEqual(agent.capital, before_cash)
        self.assertAlmostEqual(m.decision_log[0]["economic_profit"], 0)

    def test_voluntary_exit_learns_once_and_freezes_assets(self):
        m = model(cycles=5)
        agent = next(iter(m.agents))
        before = agent.net_worth()
        old = agent.action_values["exit"]
        force_action(m, "exit")
        m.run_model()
        self.assertEqual(len(m.decision_log), 1)
        self.assertTrue(m.decision_log[0]["terminal"])
        self.assertEqual(agent.exit_reason, "voluntary_exit")
        self.assertEqual(agent.net_worth(), before)
        self.assertAlmostEqual(agent.action_values["exit"], old * (1 - m.params.learning_rate))
        self.assertEqual(len(m.model_history), 6)

    def test_automatic_bankruptcy_settles_paused_agent(self):
        m = model()
        agent = next(iter(m.agents))
        agent.capital = 10
        force_action(m, "pause")
        m.run_model()
        self.assertEqual(agent.exit_reason, "bankruptcy")
        self.assertEqual(len(m.decision_log), 1)
        self.assertTrue(m.decision_log[0]["terminal"])

    def test_returns_reconcile_to_final_assets_with_shocks(self):
        m = model(n_agents=25, cycles=30, hardware_failure_prob=0.3,
                  block_reward_coins=0.2, cost_per_hash_factor=0.025)
        m.run_model()
        decisions = m.decision_log_df()
        agents = m.agent_summary_df().set_index("agent_id")
        rewards = decisions.groupby("agent_id")["economic_profit"].sum()
        np.testing.assert_allclose(rewards, agents.loc[rewards.index, "economic_profit"], atol=1e-9)
        self.assertFalse(decisions.duplicated(["agent_id", "cycle"]).any())
        self.assertTrue(np.isfinite(decisions["normalized_reward"]).all())
        for _, group in decisions.groupby("agent_id"):
            self.assertFalse(group["terminal"].iloc[:-1].any())
        count = len(m.event_log)
        m.finalize_run()
        self.assertEqual(len(m.event_log), count)


def synthetic_decisions():
    rows = []
    for run in ("train-1", "train-2"):
        for agent in range(6):
            previous = "START"
            # Same current action has different outcomes after mine and pause.
            for cycle, (action, reward) in enumerate([
                ("mine", 0), ("invest", 0.2), ("pause", 0),
                ("invest", -0.2), ("mine", 0), ("pause", 0), ("mine", 0),
            ], 1):
                rows.append({
                    "run_id": run, "agent_id": agent, "cycle": cycle,
                    "context": ("cash_ok", False), "previous_action": previous,
                    "action": action, "normalized_reward": reward,
                    "terminal": cycle == 7,
                })
                previous = action
    return pd.DataFrame(rows)


class AdviserTests(unittest.TestCase):
    def test_forward_windows_stop_at_exit_but_exclude_time_limit(self):
        rows = synthetic_decisions().query("run_id == 'train-1' and agent_id == 0").iloc[:6].copy()
        rows["normalized_reward"] = [1, 2, 3, 4, 5, 6]
        rows["terminal"] = False
        result = decision_outcomes(rows, horizon=3)
        self.assertEqual(result["forward_return"].tolist(), [6, 9, 12, 15])
        rows.loc[rows.index[-1], "terminal"] = True
        result = decision_outcomes(rows, horizon=3)
        self.assertEqual(result["forward_return"].tolist(), [6, 9, 12, 15, 11, 6])

    def test_forward_windows_do_not_mix_runs_or_agents(self):
        data = synthetic_decisions()
        result = decision_outcomes(data, horizon=10)
        for _, group in result.groupby(["run_id", "agent_id"]):
            self.assertAlmostEqual(group.iloc[0]["forward_return"], 0)
            self.assertEqual(len(group), 7)

    def test_sequence_changes_advice_and_unknown_edges_are_neutral(self):
        adviser = ProcessAdviser(horizon=1, min_support=2).fit(synthetic_decisions())
        context = ("cash_ok", False)
        self.assertGreater(adviser.lookup(context, "mine", "invest")[0], 0)
        self.assertLess(adviser.lookup(context, "pause", "invest")[0], 0)
        self.assertEqual(adviser.lookup(context, "unknown", "invest"), (0, 0))
        self.assertTrue(adviser.table["score"].between(-1, 1).all())
        self.assertEqual(adviser.training_run_ids, frozenset(("train-1", "train-2")))
        no_sequence = ProcessAdviser(horizon=1, min_support=2, use_sequence=False).fit(synthetic_decisions())
        self.assertEqual(
            no_sequence.lookup(context, "mine", "invest"),
            no_sequence.lookup(context, "pause", "invest"),
        )

    def test_unsupported_edges_do_not_produce_advice(self):
        adviser = ProcessAdviser(horizon=1, min_support=1000).fit(synthetic_decisions())
        self.assertEqual(adviser.lookup(("cash_ok", False), "mine", "invest"), (0, 0))
        one_run = synthetic_decisions().query("run_id == 'train-1'")
        adviser = ProcessAdviser(horizon=1, min_support=1).fit(one_run)
        self.assertFalse(adviser.table["supported"].any())

    def test_json_contexts_are_supported(self):
        data = synthetic_decisions()
        original = ProcessAdviser(horizon=1, min_support=2).fit(data)
        data["context"] = data["context"].map(list)
        restored = ProcessAdviser(horizon=1, min_support=2).fit(data)
        pd.testing.assert_frame_equal(original.table, restored.table)

    def test_missing_cycles_and_duplicate_decisions_are_rejected(self):
        data = synthetic_decisions()
        with self.assertRaises(ValueError):
            decision_outcomes(pd.concat([data, data.iloc[:1]]))
        with self.assertRaises(ValueError):
            decision_outcomes(data.drop(index=1))

    def test_paired_effects_use_seeds(self):
        data = pd.DataFrame([
            {"seed": seed, "variant": variant, "return": seed + bonus}
            for seed in [1, 2, 3] for variant, bonus in [("baseline", 0), ("adviser", 2)]
        ])
        effect = paired_effects(data, ["return"]).iloc[0]
        self.assertEqual(effect["mean_delta"], 2)
        self.assertEqual(effect["ci_low"], 2)
        self.assertEqual(effect["ci_high"], 2)
        self.assertEqual(effect["paired_seeds"], 3)
        with self.assertRaises(ValueError):
            paired_effects(data.iloc[:-1], ["return"])


class PairedScenarioTests(unittest.TestCase):
    def test_policy_changes_do_not_change_external_randomness(self):
        first = model(n_agents=4, cycles=8, hardware_failure_prob=0.2)
        second = model(n_agents=4, cycles=8, hardware_failure_prob=0.2)
        force_action(first, "mine")
        force_action(second, "exit")
        for _ in range(8):
            first.step()
            second.step()
            np.testing.assert_array_equal(first.failure_uniforms, second.failure_uniforms)
            np.testing.assert_array_equal(first.failure_fractions, second.failure_fractions)
            self.assertEqual(first.lottery_uniform, second.lottery_uniform)
            self.assertEqual(first.coin_price, second.coin_price)
            self.assertEqual(first.energy_grids, second.energy_grids)

    def test_zero_weight_reproduces_baseline_and_fit_stays_frozen(self):
        adviser = ProcessAdviser(horizon=1, min_support=2).fit(synthetic_decisions())
        before = adviser.table.copy(deep=True)
        baseline = model(n_agents=10, cycles=15)
        disabled = model(n_agents=10, cycles=15)
        disabled.process_adviser = adviser
        baseline.run_model()
        disabled.run_model()
        pd.testing.assert_frame_equal(baseline.agent_summary_df(), disabled.agent_summary_df())
        pd.testing.assert_frame_equal(adviser.table, before)

    def test_advice_affects_only_legal_scores(self):
        m = model(pm_weight=0.5)
        agent = next(iter(m.agents))
        old = agent.score_actions()
        adviser = ProcessAdviser()
        context = agent.decision_context()
        adviser._lookup[(context, "START", "mine")] = (0.8, 30)
        adviser._lookup[(context, "START", "leave_pool")] = (1.0, 30)
        m.process_adviser = adviser
        new = agent.score_actions()
        self.assertAlmostEqual(new["mine"] - old["mine"], 0.4)
        self.assertEqual(set(old), set(new))
        self.assertNotIn("leave_pool", new)


if __name__ == "__main__":
    unittest.main()
