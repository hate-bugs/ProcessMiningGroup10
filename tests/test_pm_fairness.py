"""Regression audit: treatment labels must confer no advantage without advice.

Uses the actual notebook model and online experiment orchestrator. No changes to
simulation rules are made by these tests; temporary output cannot overwrite runs.
"""
import copy
from dataclasses import asdict, replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from online_feedback_experiment import OnlineFeedbackExperiment, assigned_agents, online_protocol
from process_policy import TransitionProcessAdviser
from test_process_feedback import NB


def initial_state(agent):
    """All initialized agent attributes except model reference and treatment flag."""
    result = {key: copy.deepcopy(value) for key, value in vars(agent).items()
              if key not in {"model", "fleet", "_pm_weight"}}
    result["fleet"] = {"lots": [asdict(lot) for lot in agent.fleet.lots],
                       "orders": [asdict(order) for order in agent.fleet.orders],
                       "params": asdict(agent.fleet.params)}
    return result


class InitializationFairnessTests(unittest.TestCase):
    def test_assignment_changes_only_feedback_weight_for_every_test_seed(self):
        p = online_protocol()
        for seed in p["test_seeds"]:
            with self.subTest(seed=seed):
                first = NB.AdaptiveMiningModel(NB.MiningParams(n_agents=40, seed=seed))
                second = NB.AdaptiveMiningModel(NB.MiningParams(n_agents=40, seed=seed))
                first_group = assigned_agents(40, seed, False)
                second_group = assigned_agents(40, seed, True)
                before_first = [initial_state(a) for a in first.agents]
                before_second = [initial_state(a) for a in second.agents]
                self.assertEqual(before_first, before_second)
                for a in first.agents:
                    a._pm_weight = 6.0 if a.unique_id in first_group else 0.0
                for a in second.agents:
                    a._pm_weight = 6.0 if a.unique_id in second_group else 0.0
                self.assertEqual(before_first, [initial_state(a) for a in first.agents])
                self.assertEqual(before_second, [initial_state(a) for a in second.agents])
                for a, b in zip(first.agents, second.agents):
                    self.assertEqual(a.pm_weight + b.pm_weight, 6.0)
                # Every initial trait vector occurs once in each treatment across
                # complementary runs. This is stronger than equal means alone.
                treated = [initial_state(a) for a in first.agents if a.pm_weight] + [
                    initial_state(a) for a in second.agents if a.pm_weight]
                control = [initial_state(a) for a in first.agents if not a.pm_weight] + [
                    initial_state(a) for a in second.agents if not a.pm_weight]
                self.assertEqual(sorted(treated, key=lambda a: a["unique_id"]),
                                 sorted(control, key=lambda a: a["unique_id"]))

    def test_assignment_does_not_consume_simulator_randomness(self):
        model = NB.AdaptiveMiningModel(NB.MiningParams(n_agents=40, seed=19101))
        generators = [model.rng, model.environment_rng, model.failure_rng, model.lottery_rng]
        before = [copy.deepcopy(rng.bit_generator.state) for rng in generators]
        python_random = model.random.getstate()
        assigned_agents(40, 19101, False)
        assigned_agents(40, 19101, True)
        self.assertEqual(before, [rng.bit_generator.state for rng in generators])
        self.assertEqual(python_random, model.random.getstate())


class FullTrajectoryFairnessTests(unittest.TestCase):
    def assert_models_identical(self, baseline, candidate):
        # Run IDs only describe the arm; remove that single metadata column.
        for name in ["decision_log_df", "event_log_df", "model_history_df", "agent_summary_df"]:
            left = getattr(baseline, name)().drop(columns="run_id", errors="ignore")
            right = getattr(candidate, name)().drop(columns="run_id", errors="ignore")
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        for a, b in zip(baseline.agents, candidate.agents):
            self.assertEqual(a.action_values, b.action_values)
            self.assertEqual(a.fleet.lots, b.fleet.lots)
            self.assertEqual(a.fleet.orders, b.fleet.orders)
            self.assertEqual(a.capital_credit_queue, b.capital_credit_queue)
        self.assertIsNone(candidate.capital_adviser)

    def test_full_experiment_is_identical_with_zero_or_neutral_advice(self):
        # Production population, horizon, update timing and stress parameters.
        # These tests intentionally cannot be passed merely by balancing averages.
        for seed in [19101, 19104]:
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                captured = {}

                def factory(*args, **kwargs):
                    model = NB.AdaptiveMiningModel(*args, **kwargs)
                    captured["latest"] = model
                    return model

                experiment = OnlineFeedbackExperiment(factory, NB.MiningParams(), online_protocol(), directory)
                experiment.run_market(seed, "all_control", dict(weight=0.0, discount=0.0))
                baseline = captured["latest"]
                # The real adviser is repeatedly fitted, but its influence is zero.
                experiment.run_market(seed, "process", dict(weight=0.0, discount=0.0))
                self.assert_models_identical(baseline, captured["latest"])
                # Positive treatment weight and full discovery pipeline, but no
                # advice signal. The PM label alone must not change any trajectory.
                with patch.object(TransitionProcessAdviser, "lookup", return_value=(0.0, 0)):
                    for swap in [False, True]:
                        _, _, fits = experiment.run_market(seed, "process", dict(weight=6.0, discount=0.0), swap=swap)
                        self.assertEqual(len(fits), 12)
                        self.assert_models_identical(baseline, captured["latest"])

    def test_identical_actions_have_identical_accounting_with_pm_labels(self):
        params = NB.MiningParams(n_agents=12, cycles=210, seed=88231)
        baseline = NB.AdaptiveMiningModel(params)
        tagged = NB.AdaptiveMiningModel(replace(params, pm_weight=6.0))
        for model in [baseline, tagged]:
            for agent in model.agents:
                agent.choose_action = lambda: "mine"
                agent.choose_capital_action = lambda a=agent: "hold_hardware" if a.capital_review_due() else "no_review"
        # The model has no fitted adviser; all causal actions are held fixed.
        for _ in range(params.cycles):
            baseline.step()
            tagged.step()
            self.assertEqual(baseline.paid_rewards_today, tagged.paid_rewards_today)
            self.assertEqual(baseline.pool_fees_today, tagged.pool_fees_today)
            np.testing.assert_array_equal(baseline.failure_uniforms, tagged.failure_uniforms)
            np.testing.assert_array_equal(baseline.block_uniforms, tagged.block_uniforms)
        self.assert_models_identical(baseline, tagged)


if __name__ == "__main__":
    unittest.main()
