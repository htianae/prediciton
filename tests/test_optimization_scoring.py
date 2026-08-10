from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from furnace_champion.data import FEATURE_COLS, TARGET_COL, load_batch_data
from furnace_champion.optimization import (
    build_context,
    build_search_space,
    fit_feasibility_reference,
    historical_fallback,
    score_candidates,
)


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.repeat(float(self.value), len(frame))


class OptimizationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL)
        cls.context = build_context(cls.df, 86000)
        cls.space = build_search_space(cls.context["batches"])
        cls.feasibility = fit_feasibility_reference(cls.df)

    def bundle(self, fold_values):
        return {
            "model": ConstantModel(3000),
            "cv_models": [ConstantModel(value) for value in fold_values],
            "feature_cols": FEATURE_COLS,
            "target_iqr": float(self.df[TARGET_COL].quantile(.75) - self.df[TARGET_COL].quantile(.25)),
        }

    def test_context_and_search_space_follow_shared_rules(self):
        self.assertGreaterEqual(len(self.context["batches"]), 20)
        self.assertIn(self.context["weight_tolerance"], (0.05, 0.10, None))
        for column, (low, high) in self.space.bounds.items():
            self.assertAlmostEqual(low, self.context["batches"][column].quantile(.05))
            self.assertAlmostEqual(high, self.context["batches"][column].quantile(.95))

    def test_uncertainty_and_distance_increase_penalized_objective(self):
        row = self.df[FEATURE_COLS].dropna().iloc[[30]].copy()
        stable = score_candidates(self.bundle([3000, 3000]), row, self.feasibility, self.space)
        uncertain = score_candidates(self.bundle([2500, 3500]), row, self.feasibility, self.space)
        self.assertGreater(uncertain.loc[0, "penalized_objective"], stable.loc[0, "penalized_objective"])

        far = row.copy()
        far["10#熔炼炉固体料重量比例"] = 500
        far_result = score_candidates(self.bundle([3000, 3000]), far, self.feasibility, self.space)
        self.assertGreater(far_result.loc[0, "normalized_knn_distance"], stable.loc[0, "normalized_knn_distance"])
        self.assertGreater(far_result.loc[0, "penalized_objective"], stable.loc[0, "penalized_objective"])

    def test_historical_fallback_returns_four_controls(self):
        fallback = historical_fallback(self.context["batches"])
        self.assertEqual(set(fallback["recommendation"]), set(self.space.bounds))
        self.assertEqual(fallback["source"], "historical_similar_low_gas")


if __name__ == "__main__":
    unittest.main()
