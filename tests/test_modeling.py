from pathlib import Path
import unittest

import pandas as pd

from furnace_champion.data import load_batch_data, mark_target_outliers
from furnace_champion.modeling import (
    build_candidate_models,
    choose_prediction_champion,
    eligible_model_names,
    evaluate_models_time_series,
)


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = mark_target_outliers(load_batch_data(EXCEL))

    def test_candidate_models_are_exactly_the_three_approved_models(self):
        self.assertEqual(set(build_candidate_models()), {"LightGBM", "Linear", "Huber"})

    def test_time_validation_uses_only_past_to_predict_future(self):
        result = evaluate_models_time_series(self.df, n_splits=5)

        self.assertEqual(set(result["summary"]["model"]), {"LightGBM", "Linear", "Huber"})
        self.assertEqual(len(result["fold_metrics"]), 15)
        self.assertEqual(len(self.df), 292)
        self.assertEqual(int(self.df["is_high_gas_outlier"].sum()), 5)
        for train_idx, test_idx in result["splits"]:
            self.assertLess(max(train_idx), min(test_idx))
        for row in result["summary"].itertuples():
            self.assertAlmostEqual(row.selection_score, row.rmse_mean + 0.25 * row.rmse_std)

    def test_eligibility_and_tie_breaking_are_deterministic(self):
        summary = pd.DataFrame(
            [
                {"model": "LightGBM", "selection_score": 100.0, "mae_mean": 90.0},
                {"model": "Huber", "selection_score": 104.0, "mae_mean": 85.0},
                {"model": "Linear", "selection_score": 104.0, "mae_mean": 85.0},
            ]
        )
        self.assertEqual(set(eligible_model_names(summary)), {"LightGBM", "Huber", "Linear"})
        safety = pd.DataFrame(
            [
                {"model": "LightGBM", "passes_safety": False},
                {"model": "Huber", "passes_safety": True},
                {"model": "Linear", "passes_safety": True},
            ]
        )
        self.assertEqual(choose_prediction_champion(summary, safety), "Linear")


if __name__ == "__main__":
    unittest.main()
