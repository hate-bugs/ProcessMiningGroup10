"""Intervention boundaries, fixed-seed pairing and post-reference estimands."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from disruption_experiment import (CostShockPlan, ShutdownPlan, cost_shock_protocol,
    disruption_protocol, run_market, shutdown_model_class, validate_disruption_protocol)
from robustness_analysis import post_intervention_tables, seed_statistics, phase_tables, recovery_advantage
from scripts.run_disruption_experiment import prepare_output
from test_process_feedback import NB


class CostShockTests(unittest.TestCase):
    def model(self, plan=None, days=12):
        params = NB.MiningParams(n_agents=1, cycles=days, seed=117,
            hardware_failure_prob=0, imitation_rate=0, cost_per_hash_factor=.5)
        cls = shutdown_model_class(NB.AdaptiveMiningModel)
        model = cls(params, cost_shock_plan=plan)
        for agent in model.agents:
            agent.choose_action = lambda: "mine"
            agent.choose_capital_action = lambda: "no_review"
        return model

    def test_multiplier_one_exactly_preserves_baseline(self):
        base = self.model()
        neutral = self.model(CostShockPlan(4, 8, 1))
        base.run_model()
        neutral.run_model()
        for name in ["agent_summary_df", "model_history_df", "event_log_df", "decision_log_df"]:
            pd.testing.assert_frame_equal(getattr(base, name)(), getattr(neutral, name)())

    def test_actual_costs_decision_context_and_restore_on_day_540(self):
        base, shock = self.model(days=541), self.model(CostShockPlan(), days=541)
        a, b = next(iter(base.agents)), next(iter(shock.agents))
        for day in range(1, 542):
            cost_a, cost_b = a.total_costs, b.total_costs
            base.step()
            shock.step()
            expected = 2 if 360 <= day < 540 else 1
            self.assertEqual(shock.applied_cost_multiplier, expected)
            self.assertAlmostEqual(b.total_costs - cost_b, expected * (a.total_costs - cost_a))
            self.assertAlmostEqual(b.expected_cost(), expected * a.expected_cost())
            for grid in base.energy_grids:
                self.assertAlmostEqual(shock.energy_grids[grid], base.energy_grids[grid] * expected)
            self.assertEqual(shock.coin_price, base.coin_price)
            np.testing.assert_array_equal(shock.block_uniforms, base.block_uniforms)
            if day in [359, 360, 539, 540]:
                self.assertAlmostEqual(shock.decision_log[-1]["state_before"]["energy_price"],
                    base.decision_log[-1]["state_before"]["energy_price"] * expected)
        self.assertTrue(a.active and b.active)

    def test_inactive_miners_keep_original_cost_treatment(self):
        m = self.model(CostShockPlan(2, 5, 2))
        a = next(iter(m.agents))
        a.exit_market("test_exit")
        before = a.net_worth()
        m.run_model()
        self.assertEqual(a.total_costs, 0)
        self.assertEqual(a.net_worth(), before)

    def test_fixed_seed_reproducibility_and_common_cohort_definitions(self):
        p = cost_shock_protocol(True)
        args = (NB.AdaptiveMiningModel, NB.MiningParams(), p, p["seeds"][0], 2., "process", False)
        first, second = run_market(*args), run_market(*args)
        for name in ["snapshots", "agents", "daily", "cohort_daily", "fits"]:
            pd.testing.assert_frame_equal(first[name], second[name])
        neutral = run_market(NB.AdaptiveMiningModel, NB.MiningParams(), p, p["seeds"][0], 1., "process", False)
        shutdown = run_market(NB.AdaptiveMiningModel, NB.MiningParams(), disruption_protocol(True),
                              p["seeds"][0], 0., "process", False)
        cols = ["agent_id", "cohort", "status", "net_worth", "normalized_return"]
        pd.testing.assert_frame_equal(neutral["agents"][cols], shutdown["agents"][cols])
        for run in [first, neutral, shutdown]:
            self.assertEqual(run["agents"].groupby("cohort").size().to_dict(), {"control": 3, "pm": 3})
        pre = first["cohort_daily"].query("day < 6")
        baseline_pre = neutral["cohort_daily"].query("day < 6")
        common = ["seed", "day", "cohort", "active_agents", "mining_participants", "operating_cost", "mean_return_pct"]
        pd.testing.assert_frame_equal(pre[common], baseline_pre[common])

    def test_protocol_and_output_validation_preserve_history(self):
        p = cost_shock_protocol()
        validate_disruption_protocol(p)
        self.assertEqual(p["seeds"], disruption_protocol()["seeds"])
        p["cost_multipliers"] = [2]
        with self.assertRaises(ValueError):
            validate_disruption_protocol(p)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new_run"
            prepare_output(root)
            marker = root / "historical.csv"
            marker.write_text("retain me")
            with self.assertRaises(ValueError):
                prepare_output(root)
            self.assertEqual(marker.read_text(), "retain me")


class EstimandTests(unittest.TestCase):
    def fixture(self):
        p = disruption_protocol(True)
        p.update(days=10, shutdown_day=4, severities=[0, 10], seeds=[1, 2])
        rows = []
        for seed in [1, 2]:
            for severity in [0, 10]:
                for swap in [0, 1]:
                    for cohort in ["pm", "control"]:
                        # Deliberately different reference gaps between scenarios
                        # catch incorrectly subtracting only the baseline's gap.
                        reference = (100 if cohort == "pm" else 40) + seed + swap + (severity if cohort == "pm" else 0)
                        gain = (10 if cohort == "pm" else 3) if severity == 0 else (8 if cohort == "pm" else 2)
                        for day, value in [(3, reference), (10, reference + gain)]:
                            rows.append(dict(seed=seed, scenario_value=severity, family="process", swap=swap,
                                cohort=cohort, day=day, mean_return_pct=value, survival_pct=100))
        return pd.DataFrame(rows), p

    def test_post_reference_removes_preexisting_advantage(self):
        snapshots, p = self.fixture()
        returns, post, paired = post_intervention_tables(snapshots, p)
        self.assertTrue(post[post.day == 3].post_return_pp.eq(0).all())
        base = paired[(paired.scenario_value == 0) & (paired.day == 10)]
        shock = paired[(paired.scenario_value == 10) & (paired.day == 10)]
        self.assertTrue(base.post_advantage_pp.eq(7).all())
        self.assertTrue(shock.post_advantage_pp.eq(6).all())
        self.assertTrue(shock.additional_advantage_pp.eq(-1).all())
        stats = seed_statistics(paired, ["scenario_value", "day"], ["additional_advantage_pp"], 200)
        self.assertTrue(stats.paired_seeds.eq(2).all())
        row = stats[(stats.scenario_value == 10) & (stats.day == 10)].iloc[0]
        self.assertEqual((row["mean"], row.ci_low, row.ci_high, row.positive_seeds), (-1, -1, -1, 0))
        with self.assertRaises(ValueError):
            seed_statistics(pd.concat([paired, paired]), ["scenario_value", "day"], ["additional_advantage_pp"], 200)

    def test_missing_reference_or_complementary_run_is_rejected(self):
        data, p = self.fixture()
        with self.assertRaises(ValueError):
            post_intervention_tables(data[data.day != 3], p)
        with self.assertRaises(ValueError):
            post_intervention_tables(data[data.swap == 0], p)
        with self.assertRaises(ValueError):
            post_intervention_tables(data[data.seed == 1], p)

    def test_secondary_return_does_not_drop_invalid_assignment_denominators(self):
        data, p = self.fixture()
        data["return_relative_reference_wealth_pct"] = 1.0
        data.loc[(data.swap == 0) & (data.day == 10), "return_relative_reference_wealth_pct"] = np.nan
        _, post, _ = post_intervention_tables(data, p)
        self.assertTrue(post[post.day == 10].return_relative_reference_wealth_pct.isna().all())

    def test_recovery_is_paired_gap_change_not_final_wealth(self):
        rows, daily = [], []
        p = cost_shock_protocol(True)
        for seed in [1, 2]:
            for value, trajectory in [(1., [10, 20, 30]), (2., [10, 15, 28])]:
                for day, ret in zip([5, 8, 12], trajectory):
                    rows.append(dict(seed=seed, scenario_value=value, family="process", cohort="pm", day=day, mean_return_pct=ret))
                for swap in [0, 1]:
                    for day in range(1, 13):
                        daily.append(dict(seed=seed, scenario_value=value, family="process", cohort="pm", day=day,
                            swap=swap, active_agents=3, mining_participants=2, operating_cost=1,
                            net_mining_profit=1, normalized_operating_profit_pp=.1))
        phases, recovery = phase_tables(pd.DataFrame(rows), pd.DataFrame(daily), p)
        shock = recovery[recovery.scenario_value == 2]
        self.assertTrue(shock.gap_at_shock_end_pp.eq(-5).all())
        self.assertTrue(shock.final_gap_pp.eq(-2).all())
        self.assertTrue(shock.recovery_gap_change_pp.eq(3).all())
        self.assertEqual(set(phases.phase), {"before", "during", "after"})

    def test_guided_recovery_comparison_preserves_seed_pairing(self):
        rows = [dict(seed=seed, scenario_value=2, family="process", cohort=cohort,
                     recovery_gap_change_pp=value)
                for seed, guided, control in [(1, 30, 20), (2, 2, 12)]
                for cohort, value in [("pm", guided), ("control", control)]]
        paired = recovery_advantage(pd.DataFrame(rows))
        self.assertEqual(paired.guided_minus_control_recovery_gap_change_pp.tolist(), [10, -10])
        effects = seed_statistics(paired, ["scenario_value"], ["guided_minus_control_recovery_gap_change_pp"], 200)
        self.assertEqual(int(effects.iloc[0].paired_seeds), 2)
        self.assertEqual(int(effects.iloc[0].positive_seeds), 1)
        self.assertEqual(float(effects.iloc[0]["mean"]), 0)


if __name__ == "__main__":
    unittest.main()
