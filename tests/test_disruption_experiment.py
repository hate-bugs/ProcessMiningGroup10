"""Forced-closure accounting, no-op equivalence, and paired sweep checks."""
from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from disruption_experiment import (
    ShutdownPlan, disruption_protocol, run_market, shutdown_model_class,
    summarize, voluntary_history,
)
from process_policy import TransitionProcessAdviser
from test_process_feedback import NB


class ShutdownTests(unittest.TestCase):
    def params(self, **kwargs):
        return NB.MiningParams(**dict(dict(n_agents=8, cycles=8, seed=341,
            hardware_failure_prob=0, cost_per_hash_factor=0, imitation_rate=0), **kwargs))

    def forced_model(self, percentage=50, day=4, **kwargs):
        cls = shutdown_model_class(NB.AdaptiveMiningModel)
        m = cls(self.params(**kwargs), shutdown_plan=ShutdownPlan(day, percentage, 341))
        for a in m.agents:
            a.choose_action = lambda: "mine"
            a.choose_capital_action = lambda: "no_review"
        return m

    def test_zero_percent_preserves_full_baseline_trajectory(self):
        params = self.params()
        baseline = NB.AdaptiveMiningModel(params)
        adapted = shutdown_model_class(NB.AdaptiveMiningModel)(params, shutdown_plan=ShutdownPlan(4, 0, 341))
        baseline.run_model()
        adapted.run_model()
        pd.testing.assert_frame_equal(baseline.agent_summary_df(), adapted.agent_summary_df())
        pd.testing.assert_frame_equal(baseline.decision_log_df(), adapted.decision_log_df())
        pd.testing.assert_frame_equal(baseline.event_log_df(), adapted.event_log_df())
        pd.testing.assert_frame_equal(baseline.model_history_df(), adapted.model_history_df())

    def test_no_effect_before_intervention_and_exogenous_streams_match(self):
        baseline, adapted = self.forced_model(0), self.forced_model(50)
        for _ in range(3):
            baseline.step()
            adapted.step()
        pd.testing.assert_frame_equal(baseline.decision_log_df(), adapted.decision_log_df())
        for _ in range(5):
            baseline.step()
            adapted.step()
            self.assertEqual(baseline.coin_price, adapted.coin_price)
            self.assertEqual(baseline.energy_grids, adapted.energy_grids)
            np.testing.assert_array_equal(baseline.block_uniforms, adapted.block_uniforms)
            np.testing.assert_array_equal(baseline.failure_uniforms, adapted.failure_uniforms)

    def test_shutdown_is_permanent_and_full_shutdown_stops_production(self):
        model = self.forced_model(100)
        model.run_model()
        self.assertEqual(model.shutdown_audit["shutdown_count"], 8)
        agents = model.agent_summary_df()
        self.assertTrue(agents.exit_reason.eq("external_shutdown").all())
        daily = model.model_history_df().query("cycle >= 4")
        self.assertTrue(daily.active_agents.eq(0).all())
        self.assertTrue(daily.mined_blocks.eq(0).all())
        self.assertTrue(daily.paid_mining_rewards.eq(0).all())
        self.assertEqual(model.decision_log_df().cycle.max(), 4)
        self.assertTrue(agents.hash_power.eq(0).all())

    def test_shutdown_cancellation_reconciles_and_does_not_train_action_values(self):
        model = self.forced_model(100, day=1, n_agents=1)
        a = next(iter(model.agents))
        a.capital -= 100
        a.fleet.order(1, 100, 0)
        before = a.net_worth()
        values = a.action_values.copy()
        model.step()
        row = model.decision_log_df().iloc[0]
        self.assertAlmostEqual(a.net_worth(), before - 10)
        self.assertAlmostEqual(row.economic_profit, -10)
        self.assertTrue(row.externally_forced)
        self.assertEqual(a.action_values, values)
        self.assertEqual(a.total_rewards, 0)
        self.assertEqual(a.total_costs, 0)
        self.assertEqual(len(a.fleet.orders), 0)
        self.assertGreater(a.max_drawdown, 0)

    def test_forced_records_close_case_but_are_not_adviser_actions(self):
        model = self.forced_model(50)
        model.run_model()
        data = model.decision_log_df()
        for agent in model.agents:
            case = data[data.agent_id == agent.unique_id]
            self.assertAlmostEqual(case.normalized_reward.sum(),
                (agent.net_worth() - agent.initial_net_worth) / agent.initial_net_worth)
            np.testing.assert_array_equal(np.diff(case.cycle), 1)
        history = voluntary_history(model, 8, 8)
        self.assertFalse(history.action.eq("external_shutdown").any())
        adviser = TransitionProcessAdviser(min_support=1, min_agents=1).fit(history)
        self.assertFalse(adviser.table.action.eq("external_shutdown").any())
        selected = set(model.shutdown_audit["shutdown_ids"])
        censored = history[history.agent_id.isin(selected)].groupby("agent_id").tail(1)
        self.assertTrue(censored.cycle.eq(3).all())
        self.assertFalse(censored.terminal.any())

    def test_nested_targets_and_active_denominator(self):
        model = self.forced_model()
        next(iter(model.agents)).exit_market("previous_exit")
        eligible, small = ShutdownPlan(4, 5, 341).targets(model)
        _, medium = ShutdownPlan(4, 10, 341).targets(model)
        _, large = ShutdownPlan(4, 50, 341).targets(model)
        self.assertEqual(len(eligible), 7)
        self.assertEqual(len(small), 1)
        self.assertTrue(small <= medium <= large)
        self.assertNotIn(1, large)
        self.assertEqual(ShutdownPlan(4, 100, 341).targets(model)[1], set(eligible))
        for a in model.agents:
            a.exit_market("previous_exit")
        self.assertEqual(ShutdownPlan(4, 100, 341).targets(model), ([], set()))


class SweepTests(unittest.TestCase):
    def test_seeds_are_new_and_protocol_covers_extremes(self):
        p = disruption_protocol()
        self.assertTrue({0, 5, 10, 100}.issubset(p["severities"]))
        self.assertFalse(set(p["seeds"]) & set(range(19101, 19109)))
        self.assertFalse(set(p["seeds"]) & set(disruption_protocol(True)["seeds"]))

    def test_paired_summary_averages_swaps_before_bootstrap(self):
        p = disruption_protocol(True)
        p["severities"] = [0, 100]
        results = []
        for seed in p["seeds"]:
            for percentage in p["severities"]:
                for family, swap in [("all_control", False), ("process", False), ("process", True)]:
                    results.append(run_market(NB.AdaptiveMiningModel, NB.MiningParams(), p,
                                              seed, percentage, family, swap))
        seeds, effects = summarize(pd.concat([r["snapshots"] for r in results]), p)
        self.assertEqual(len(seeds), 4)
        self.assertTrue(effects.paired_seeds.eq(2).all())
        zero = effects[(effects.severity_pct == 0) & effects.metric.isin([
            "advantage_change_vs_no_shutdown", "market_change_vs_no_shutdown"])]
        np.testing.assert_allclose(zero[["mean_delta", "ci_low", "ci_high"]], 0, atol=1e-12)
        full = effects[(effects.severity_pct == 100) & (effects.metric == "survival_pct")]
        self.assertEqual(float(full.iloc[0].mean_delta), 0)


if __name__ == "__main__":
    unittest.main()
