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
        cost_per_hash_factor=0, daily_reward_coins=0, imitation_rate=0, initial_hardware_age_days=0,
    )
    kwargs.update(overrides)
    return NB.AdaptiveMiningModel(NB.MiningParams(**kwargs))


def force_action(m, action):
    for agent in m.agents:
        agent.choose_action = lambda: action
        agent.choose_capital_action = lambda agent=agent: "hold_hardware" if agent.capital_review_due() else "no_review"


class AccountingTests(unittest.TestCase):
    def test_pause_settles_once_and_depreciates_without_fake_operating_reward(self):
        m = model()
        agent = next(iter(m.agents))
        old = agent.action_values["pause"]
        force_action(m, "pause")
        m.run_model()
        self.assertEqual(len(m.decision_log), 1)
        self.assertLess(m.decision_log[0]["economic_profit"], 0)
        self.assertEqual(m.decision_log[0]["mining_profit"], 0)
        self.assertAlmostEqual(agent.action_values["pause"], old * (1 - m.params.learning_rate))
        self.assertFalse(m.decision_log[0]["terminal"])

    def test_repair_preserves_age_and_recovers_asset_value(self):
        m = model()
        agent = next(iter(m.agents))
        agent.fleet.damage(0.2)
        before = agent.net_worth()
        age = agent.fleet.lots[0].age_days
        repaired, cost = agent.fleet.repair(1000)
        agent.capital -= cost
        self.assertGreater(repaired, 0)
        self.assertAlmostEqual(agent.net_worth(), before)
        self.assertEqual(agent.fleet.lots[0].age_days, age)

    def test_uneconomic_repair_cost_reaches_operating_learner(self):
        m = model(repair_cost_per_hash=200)
        agent = next(iter(m.agents))
        agent.fleet.damage(0.1)
        old = agent.action_values["repair"]
        force_action(m, "repair")
        m.run_model()
        reward = m.decision_log[0]["learning_reward"]
        self.assertLess(reward, 0)
        self.assertAlmostEqual(agent.action_values["repair"], old + m.params.learning_rate * (reward - old))

    def test_sale_exchanges_assets_for_cash_at_age_dependent_value(self):
        m = model()
        agent = next(iter(m.agents))
        before = agent.net_worth()
        old_hash = agent.hash_power
        sold, cash = agent.fleet.sell(old_hash * 0.2)
        agent.capital += cash
        self.assertAlmostEqual(agent.net_worth(), before)
        self.assertAlmostEqual(agent.hash_power, old_hash - sold)

    def test_orders_do_not_mine_before_delivery(self):
        m = model(delivery_delay_days=3)
        agent = next(iter(m.agents))
        old_hash = agent.hash_power
        before = agent.net_worth()
        agent.capital -= 80
        agent.fleet.order(1, 80, day=1)
        self.assertAlmostEqual(agent.net_worth(), before)
        self.assertEqual(agent.hash_power, old_hash)
        self.assertEqual(agent.fleet.advance_day(3)[0], 0)
        self.assertEqual(agent.fleet.advance_day(4)[0], 1)
        self.assertEqual(agent.hash_power, old_hash + 1)
        self.assertEqual(agent.fleet.pending_hash, 0)

    def test_old_hardware_retires_and_pays_residual_value(self):
        m = model(hardware_lifetime_days=2, initial_hardware_age_days=1)
        agent = next(iter(m.agents))
        expected_salvage = agent.hash_power * agent.fleet.lots[0].unit_cost * m.params.residual_value_fraction
        cash = agent.capital
        force_action(m, "pause")
        m.run_model()
        self.assertEqual(agent.hash_power, 0)
        self.assertAlmostEqual(agent.capital, cash + expected_salvage)
        self.assertTrue(any(e["activity"] == "Hardware retired" for e in m.event_log))

    def test_capital_review_calendar_and_cooldown(self):
        m = model(cycles=100)
        agent = next(iter(m.agents))
        for day in range(1, 101):
            m.steps = day
            self.assertEqual(agent.capital_review_due(), day in [1, 31, 61, 91])
        agent.last_capital_trade_day = 1
        m.steps = 31
        self.assertEqual(agent.legal_capital_actions(), ["hold_hardware"])
        m.steps = 91
        self.assertIn("sell_hardware", agent.legal_capital_actions())
        self.assertNotIn("invest", agent.legal_actions())
        self.assertNotIn("sell_hardware", agent.legal_actions())

    def test_capital_learning_waits_for_observed_horizon(self):
        m = model(cycles=180)
        agent = next(iter(m.agents))
        force_action(m, "pause")
        for _ in range(179):
            m.step()
        self.assertEqual(agent.capital_updates, 0)
        m.step()
        self.assertEqual(agent.capital_updates, 1)
        self.assertGreater(len(agent.capital_credit_queue), 0)
        m.finalize_run()
        self.assertEqual(agent.capital_updates, 1)  # no labels from truncated futures

    def test_exit_liquidates_and_cancels_orders(self):
        m = model(cycles=4)
        agent = next(iter(m.agents))
        agent.capital -= 80
        agent.fleet.order(1, 80, day=0)
        force_action(m, "exit")
        m.run_model()
        self.assertEqual(len(m.decision_log), 1)
        self.assertTrue(m.decision_log[0]["terminal"])
        self.assertEqual(agent.hash_power, 0)
        self.assertEqual(agent.fleet.pending_hash, 0)
        self.assertEqual(agent.net_worth(), agent.capital)
        self.assertLess(m.decision_log[0]["economic_profit"], -7.99)

    def test_automatic_exit_includes_order_cancellation_in_economic_reward(self):
        m = model()
        agent = next(iter(m.agents))
        agent.capital = 10
        agent.fleet.order(1, 80, day=1)
        force_action(m, "pause")
        m.run_model()
        row = m.decision_log[0]
        self.assertEqual(agent.exit_reason, "bankruptcy")
        self.assertAlmostEqual(row["economic_profit"], row["net_worth_after"] - row["net_worth_before"])
        self.assertLess(row["learning_reward"], 0)
        self.assertEqual(agent.fleet.pending_hash, 0)

    def test_returns_reconcile_with_ageing_orders_and_exit(self):
        m = model(n_agents=15, cycles=210, hardware_failure_prob=0.05,
                  daily_reward_coins=1, cost_per_hash_factor=0.5)
        m.run_model()
        decisions = m.decision_log_df()
        agents = m.agent_summary_df().set_index("agent_id")
        rewards = decisions.groupby("agent_id")["economic_profit"].sum()
        np.testing.assert_allclose(rewards, agents.loc[rewards.index, "economic_profit"], atol=1e-8)
        self.assertFalse(decisions.duplicated(["agent_id", "cycle"]).any())
        reviews = decisions[decisions.capital_action != "no_review"]
        for _, group in reviews.groupby("agent_id"):
            self.assertTrue((np.diff(group.cycle) == 30).all())
        trades = decisions[decisions.capital_action.isin(["invest", "sell_hardware"])]
        for _, group in trades.groupby("agent_id"):
            self.assertTrue((np.diff(group.cycle) >= 90).all())
        for _, group in decisions.groupby("agent_id"):
            self.assertFalse(group["terminal"].iloc[:-1].any())

    def test_blocks_are_batched_and_energy_is_charged_once_daily(self):
        m = model(n_agents=3, daily_reward_coins=1, cost_per_hash_factor=0.5)
        force_action(m, "mine")
        m.run_model()
        self.assertEqual(len(m.decision_log), 3)
        self.assertGreater(m.blocks_today, 100)
        self.assertEqual(m.mined_blocks_today, m.blocks_today)
        expected = m.blocks_today * m.params.daily_reward_coins * m.coin_price / m.params.expected_blocks_per_day
        self.assertAlmostEqual(m.paid_rewards_today + m.pool_fees_today, expected)
        for agent in m.agents:
            self.assertEqual(agent.times_mined, 1)
            self.assertAlmostEqual(agent.total_costs, agent.expected_cost())

    def test_zero_block_day_still_charges_energy(self):
        m = model(cost_per_hash_factor=0.5)
        agent = next(iter(m.agents))
        agent.last_action = "mine"
        m.blocks_today = 0
        m.block_uniforms = np.array([])
        m.run_mining_round()
        self.assertEqual(m.paid_rewards_today, 0)
        self.assertGreater(agent.total_costs, 0)

    def test_pool_reward_budget_is_conserved(self):
        m = model(n_agents=4, daily_reward_coins=1)
        force_action(m, "mine")
        for agent in m.agents:
            agent.pool_id = 1
        m.run_model()
        self.assertAlmostEqual(m.pool_fees_today, m.gross_rewards_today * m.params.pool_fee)
        self.assertAlmostEqual(m.paid_rewards_today + m.pool_fees_today, m.gross_rewards_today)

    def test_expected_daily_issuance_does_not_scale_with_block_resolution(self):
        for blocks in (1, 144, 288):
            m = model(daily_reward_coins=1, expected_blocks_per_day=blocks)
            m.blocks_today = blocks
            m.block_uniforms = np.linspace(0, 1, blocks, endpoint=False)
            m.run_mining_round()
            self.assertAlmostEqual(m.gross_rewards_today, m.coin_price)


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

    def test_capital_windows_include_days_between_reviews(self):
        data = synthetic_decisions()
        data["normalized_reward"] = 1.0
        data["capital_action"] = np.where(data.cycle.isin([1, 4, 7]), "hold_hardware", "no_review")
        data["capital_context"] = data["context"]
        data["previous_capital_action"] = np.where(data.cycle == 1, "START", "hold_hardware")
        data["initial_net_worth"] = 100.0
        data["net_worth_before"] = 100.0
        data["capital_wealth_before"] = 102.0
        adviser = ProcessAdviser(horizon=3, decision_kind="capital").fit(data)
        self.assertEqual(adviser.table["support"].sum(), 36)
        self.assertNotIn("no_review", set(adviser.table.action))
        first = adviser.table[adviser.table.history == "START"]
        self.assertTrue(np.allclose(first.mean_return, 2.98))

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
            self.assertEqual(first.blocks_today, second.blocks_today)
            np.testing.assert_array_equal(first.block_uniforms, second.block_uniforms)
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
