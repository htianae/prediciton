from pathlib import Path
import unittest

import numpy as np

from advanced_furnace_ml.data import chronological_dev_lock_split, load_batch_data
from advanced_furnace_ml.experiment import (
    audit_frozen_models,
    evaluate_candidate,
    run_model_matrix,
)
from advanced_furnace_ml.models import RouteRegressor, build_base_models
from advanced_furnace_ml.uncertainty import conformal_radius, prediction_interval
from advanced_furnace_ml.validation import large_window_splits


EXCEL = Path(__file__).parents[2] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL)
        cls.dev, cls.locked = chronological_dev_lock_split(cls.df)
        cls.splits = large_window_splits(len(cls.dev))

    def test_oof_and_conformal_use_development_validation_only(self):
        estimator = RouteRegressor(build_base_models()["Ridge"], "direct")
        result = evaluate_candidate("Ridge__direct", estimator, "direct", self.dev, self.splits)
        self.assertTrue(np.isnan(result.oof_predictions[:148]).all())
        self.assertTrue(np.isfinite(result.oof_predictions[148:]).all())
        self.assertEqual(len(result.fold_models), 3)
        radius = conformal_radius(self.dev.iloc[:, -1].to_numpy(), result.oof_predictions)
        self.assertGreater(radius, 0)
        low, high = prediction_interval(np.array([1000.0]), radius)
        self.assertLess(low[0], high[0])

    def test_selection_freezes_before_locked_audit(self):
        experiment = run_model_matrix(
            self.dev,
            self.splits,
            model_names=("Ridge", "LightGBM"),
            routes=("direct",),
        )
        self.assertEqual(set(experiment.tree_tuning_summary["model"]), {"LightGBM"})
        self.assertEqual(int(experiment.tree_tuning_summary["selected"].sum()), 1)
        selected = experiment.selected_name
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            audit_frozen_models(experiment, self.locked)
        experiment.freeze()
        audit = audit_frozen_models(experiment, self.locked)
        self.assertEqual(len(audit), 3)
        self.assertEqual(experiment.selected_name, selected)
        self.assertIn("OOFEnsemble", set(audit["candidate"]))


if __name__ == "__main__":
    unittest.main()
