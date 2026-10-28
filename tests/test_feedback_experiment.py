"""Scientific workflow checks against the actual notebook simulator."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from feedback_experiment import FeedbackExperiment, default_protocol, select_candidate, validate_protocol
from test_process_feedback import NB


class ProtocolTests(unittest.TestCase):
    def test_seed_leakage_and_repeated_training_scenarios_are_rejected(self):
        for target in ["validation_seeds", "evaluation_seeds"]:
            p = default_protocol(smoke=True)
            p[target][0] = p["training_seeds"][0][0]
            with self.assertRaises(ValueError):
                validate_protocol(p)
        p = default_protocol(smoke=True)
        p["training_seeds"][1][0] = p["training_seeds"][0][0]
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_selection_uses_seed_mean_not_best_seed_and_breaks_ties_stably(self):
        rows = pd.DataFrame({"candidate": [0, 0, 1, 1, 2, 2],
                             "mean_return_pct": [3, 3, 10, -5, 3, 3]})
        self.assertEqual(select_candidate(rows), 0)

    def test_paper_protocol_has_more_rounds_and_untouched_test_seeds(self):
        p = default_protocol()
        validate_protocol(p)
        self.assertEqual(p["rounds"], 6)
        self.assertEqual(len(p["evaluation_seeds"]), 8)
        self.assertEqual(p["advice_candidates"][0]["pm_weight"], 0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.protocol = default_protocol(smoke=True)
        cls.experiment = FeedbackExperiment(NB.AdaptiveMiningModel, NB.MiningParams(),
                                            cls.protocol, cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.experiment.tune()
            cls.experiment.train()
            cls.before = { (arm, r, k): adviser.table.copy(deep=True)
                           for arm, checkpoints in cls.experiment.checkpoints.items()
                           for r, advisers in checkpoints.items()
                           for k, adviser in enumerate(advisers) }
            cls.experiment.evaluate()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_evaluation_never_reaches_fitting_or_selection(self):
        exp = self.experiment
        self.assertEqual(set(exp.tuning_results.seed), set(self.protocol["validation_seeds"]))
        self.assertEqual(set(exp.results.seed), set(self.protocol["evaluation_seeds"]))
        first_test = next(i for i, r in enumerate(exp.audit_rows) if r["stage"].startswith("test-"))
        self.assertTrue(all(r["stage"].startswith("test-") for r in exp.audit_rows[first_test:]))
        for arm, checkpoints in exp.checkpoints.items():
            for round_number, advisers in checkpoints.items():
                for kind, adviser in enumerate(advisers):
                    pd.testing.assert_frame_equal(adviser.table, self.before[arm, round_number, kind])
                    expected_seeds = {s for batch in self.protocol["training_seeds"][:round_number] for s in batch}
                    actual_seeds = {int(run.rsplit("-", 1)[-1]) for run in adviser.training_run_ids}
                    self.assertEqual(actual_seeds, expected_seeds)

    def test_all_checkpoints_and_complete_calendar_are_reported(self):
        exp = self.experiment
        self.assertEqual(set(exp.results.variant), {"baseline", "pm_round_1", "pm_round_2", "no_sequence"})
        self.assertEqual(len(exp.results), 2 * 4 * 2)
        self.assertFalse(exp.results.duplicated(["seed", "variant", "horizon_days"]).any())
        self.assertTrue((exp.trajectories.groupby(["seed", "variant"]).size() == 12).all())
        self.assertIn("SMOKE TEST ONLY", exp.conclusion())
        self.assertTrue((Path(self.temp.name) / "parameter_search.csv").is_file())

    def test_smaller_population_preserves_initial_reward_budget_per_miner(self):
        self.assertAlmostEqual(self.experiment.params.daily_reward_coins / self.protocol["n_agents"], 1 / 150)

    def test_no_test_based_retuning_or_repeat_evaluation(self):
        with self.assertRaises(RuntimeError):
            self.experiment.tune()
        with self.assertRaises(RuntimeError):
            self.experiment.evaluate()

    def test_active_advice_is_tested_even_when_validation_favors_control(self):
        # Synthetic validation scores exercise the negative-result branch; these
        # temporary records never enter the actual research experiment.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.experiment.tuning_results.copy()
            rows.loc[rows.pm_weight > 0, "mean_return_pct"] = -1000
            rows.to_csv(root / "parameter_search.csv", index=False)
            (root / "experiment_config.json").write_text(json.dumps(self.experiment.manifest))
            exp = FeedbackExperiment(NB.AdaptiveMiningModel, NB.MiningParams(), self.protocol, root / "rerun")
            with contextlib.redirect_stdout(io.StringIO()):
                exp.tune(validation_cache=root / "parameter_search.csv")
            self.assertGreater(exp.selection["process"]["pm_weight"], 0)
            self.assertGreater(exp.selection["no_sequence"]["pm_weight"], 0)
            self.assertEqual(len(exp.tuning_results), len(rows))
            self.assertIn("validation_cache_sha256", exp.manifest)


if __name__ == "__main__":
    unittest.main()
