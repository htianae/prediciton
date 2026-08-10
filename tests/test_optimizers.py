from pathlib import Path
import unittest

import numpy as np

from furnace_champion.data import FEATURE_COLS, TARGET_COL, load_batch_data
from furnace_champion.optimization import (
    build_context,
    build_search_space,
    compare_optimizers,
    fit_feasibility_reference,
    genetic_search,
    random_search,
)


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class SmoothModel:
    def predict(self, frame):
        return 2500 + 0.01 * frame[FEATURE_COLS[0]].to_numpy() + frame[FEATURE_COLS[1]].to_numpy()


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL)
        cls.context = build_context(cls.df, 86000)
        cls.space = build_search_space(cls.context["batches"])
        cls.feasibility = fit_feasibility_reference(cls.df)
        cls.bundle = {
            "model": SmoothModel(),
            "cv_models": [SmoothModel(), SmoothModel()],
            "feature_cols": FEATURE_COLS,
            "target_iqr": float(cls.df[TARGET_COL].quantile(.75) - cls.df[TARGET_COL].quantile(.25)),
        }

    def test_random_and_ga_share_bounds_budget_and_are_deterministic(self):
        random_a = random_search(self.bundle, self.context, self.space, self.feasibility, budget=5000, seed=7)
        random_b = random_search(self.bundle, self.context, self.space, self.feasibility, budget=5000, seed=7)
        ga = genetic_search(self.bundle, self.context, self.space, self.feasibility, budget=5000, seed=7)
        self.assertEqual(len(random_a), 5000)
        self.assertEqual(len(ga), 5000)
        self.assertTrue(random_a.equals(random_b))
        self.assertTrue(np.allclose(ga["熔炼炉B当前批次炉门打开次数_PLC"] % 1, 0))
        self.assertEqual(random_a.attrs["search_bounds"], ga.attrs["search_bounds"])
        for column, (low, high) in self.space.bounds.items():
            self.assertTrue(random_a[column].between(low, high).all(), column)
            self.assertTrue(ga[column].between(low, high).all(), column)

    def test_half_percent_tie_prefers_random_search(self):
        comparison = compare_optimizers(
            self.bundle,
            self.context,
            self.space,
            self.feasibility,
            seeds=[1],
            budget=200,
            tie_ratio=1.0,
        )
        self.assertEqual(comparison["optimizer_name"], "random_search")

    def test_comparison_can_release_full_candidate_tables(self):
        comparison = compare_optimizers(
            self.bundle,
            self.context,
            self.space,
            self.feasibility,
            seeds=[1],
            budget=200,
            keep_results=False,
        )
        self.assertNotIn("results", comparison)
        self.assertEqual(len(comparison["best_candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
