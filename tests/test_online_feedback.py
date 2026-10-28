"""Process discovery, data-derived advice, assignment fairness and time isolation."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from process_policy import TransitionProcessAdviser
from online_feedback_experiment import OnlineFeedbackExperiment, assigned_agents, online_protocol, validate_protocol
from test_process_feedback import NB


def traces():
    rows = []
    for agent in range(3):
        previous = "START"
        for day, action in enumerate(["alpha", "beta", "alpha", "alpha", "beta", "beta", "alpha", "beta"], 1):
            rows.append(dict(run_id="synthetic", agent_id=agent, cycle=day, context=("s",),
                             previous_action=previous, action=action, terminal=day == 8,
                             learning_reward=1.0 if action == "alpha" else -1.0))
            previous = action
    return pd.DataFrame(rows)


class DiscoveredPolicyTests(unittest.TestCase):
    def adviser(self, **kwargs):
        return TransitionProcessAdviser(min_support=1, min_agents=2, **kwargs)

    def test_observed_rewards_determine_and_reverse_recommendations(self):
        data = traces()
        first = self.adviser(discount=0).fit(data)
        self.assertGreater(first.lookup(("s",), "alpha", "alpha")[0], first.lookup(("s",), "alpha", "beta")[0])
        data.learning_reward *= -1
        second = self.adviser(discount=0).fit(data)
        self.assertLess(second.lookup(("s",), "alpha", "alpha")[0], second.lookup(("s",), "alpha", "beta")[0])

    def test_activity_names_have_no_built_in_meaning(self):
        data = traces()
        first = self.adviser(discount=.9).fit(data)
        mapping = {"alpha": "opaque_493", "beta": "opaque_805", "START": "START"}
        data.action = data.action.map(mapping)
        data.previous_action = data.previous_action.map(mapping)
        second = self.adviser(discount=.9).fit(data)
        for history in mapping:
            for action in ["alpha", "beta"]:
                self.assertEqual(first.lookup(("s",), history, action), second.lookup(("s",), mapping[history], mapping[action]))

    def test_discovered_future_transitions_change_the_plan(self):
        rows = []
        for action, immediate, future in [("alpha", 0., 1.), ("beta", .2, 0.)]:
            for agent in range(3):
                case = f"{action}-{agent}"
                rows.extend([
                    dict(run_id=case, agent_id=agent, cycle=1, context=("start",), previous_action="START",
                         action=action, terminal=False, learning_reward=immediate),
                    dict(run_id=case, agent_id=agent, cycle=2, context=(action,), previous_action=action,
                         action="finish", terminal=True, learning_reward=future),
                ])
        data = pd.DataFrame(rows)
        immediate = self.adviser(discount=0).fit(data)
        planned = self.adviser(discount=.9).fit(data)
        self.assertLess(immediate.lookup(("start",), "START", "alpha")[0], 0)
        self.assertGreater(planned.lookup(("start",), "START", "alpha")[0], 0)
        sums = planned.transition_table.groupby("edge_id").probability.sum()
        np.testing.assert_allclose(sums, 1)

    def test_empty_and_unseen_histories_do_not_supply_advice(self):
        adviser = self.adviser().fit(traces().iloc[:0])
        self.assertEqual(adviser.lookup(("s",), "alpha", "alpha"), (0, 0))
        adviser.fit(traces())
        self.assertEqual(adviser.lookup(("unseen",), "alpha", "alpha"), (0, 0))
        sparse = TransitionProcessAdviser(min_support=100).fit(traces())
        self.assertEqual(sparse.lookup(("s",), "alpha", "alpha"), (0, 0))

    def test_last_unfinished_transition_is_censored_per_case(self):
        data = traces()
        data.terminal = False
        adviser = self.adviser().fit(data)
        self.assertEqual(adviser.table.support.sum(), len(data) - 3)
        with self.assertRaises(ValueError):
            self.adviser().fit(data.drop(index=2))
        with self.assertRaises(ValueError):
            self.adviser().fit(pd.concat([data, data.iloc[:1]]))

    def test_no_sequence_ablation_ignores_previous_activity_only(self):
        adviser = self.adviser(use_sequence=False).fit(traces())
        self.assertEqual(adviser.lookup(("s",), "alpha", "alpha"), adviser.lookup(("s",), "arbitrary", "alpha"))


class AssignmentTests(unittest.TestCase):
    def test_balanced_complementary_assignment(self):
        first = assigned_agents(40, 12, False)
        second = assigned_agents(40, 12, True)
        self.assertEqual(len(first), 20)
        self.assertFalse(first & second)
        self.assertEqual(first | second, set(range(1, 41)))

    def test_control_preserves_all_original_scores(self):
        baseline = NB.AdaptiveMiningModel(NB.MiningParams(n_agents=4, seed=33))
        mixed = NB.AdaptiveMiningModel(NB.MiningParams(n_agents=4, seed=33, pm_weight=6))
        a, b = next(iter(baseline.agents)), next(iter(mixed.agents))
        b._pm_weight = 0
        mixed.process_adviser = TransitionProcessAdviser()
        mixed.process_adviser._lookup[(b.decision_context(), "START", "mine")] = (1.0, 100)
        self.assertEqual(a.score_actions(), b.score_actions())
        b._pm_weight = 6
        self.assertAlmostEqual(b.score_actions()["mine"] - a.score_actions()["mine"], 6)

    def test_validation_test_and_smoke_seeds_are_separate(self):
        full, smoke = online_protocol(), online_protocol(True)
        validate_protocol(full)
        self.assertFalse(set(full["test_seeds"]) & set(smoke["test_seeds"]))
        full["test_seeds"][0] = full["validation_seeds"][0]
        with self.assertRaises(ValueError):
            validate_protocol(full)


class OnlinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.exp = OnlineFeedbackExperiment(NB.AdaptiveMiningModel, NB.MiningParams(), online_protocol(True), cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.exp.tune()
            cls.exp.evaluate()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_advice_cannot_use_future_events(self):
        audit = self.exp.fit_audit
        self.assertTrue((audit.max_observed_day <= audit.update_day).all())
        self.assertTrue((audit.first_affected_day == audit.update_day + 1).all())
        warmup = self.exp.seed_effects.query("feedback_round == 0")
        np.testing.assert_allclose(warmup.mean_return_pct, 0, atol=1e-12)
        first_test = next(i for i, row in enumerate(self.exp.audit) if row["stage"] == "test")
        self.assertTrue(all(row["stage"] == "test" for row in self.exp.audit[first_test:]))

    def test_controls_receive_no_logged_advice(self):
        evidence = next((Path(self.temp.name) / "evidence").glob("test-process-*_decisions.jsonl.gz"))
        data = pd.read_json(evidence, lines=True)
        self.assertTrue((data.loc[~data.treated, "pm_bonus"] == 0).all())
        self.assertTrue((data.loc[~data.treated, "pm_support"] == 0).all())

    def test_same_agent_cases_span_feedback_boundaries(self):
        evidence = next((Path(self.temp.name) / "evidence").glob("test-process-*_decisions.jsonl.gz"))
        data = pd.read_json(evidence, lines=True)
        for _, case in data.groupby("agent_id"):
            self.assertEqual(case.run_id.nunique(), 1)
            np.testing.assert_array_equal(np.diff(case.cycle), 1)
        self.assertEqual(data.agent_id.nunique(), self.exp.protocol["n_agents"])

    def test_assignments_are_not_counted_as_independent_seeds(self):
        self.assertTrue((self.exp.effects.paired_seeds == 2).all())
        self.assertEqual(set(self.exp.effects.feedback_round), {0, 1, 2})
        self.assertEqual(set(self.exp.results.seed), set(self.exp.protocol["test_seeds"]))
        with self.assertRaises(RuntimeError):
            self.exp.evaluate()


if __name__ == "__main__":
    unittest.main()
