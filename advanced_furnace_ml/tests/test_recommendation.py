from pathlib import Path
import unittest

import numpy as np

from advanced_furnace_ml.data import (
    DOOR_COUNT_COL,
    FEATURE_COLS,
    SOLID_COL,
    WAIT_COL,
    DOOR_DURATION_COL,
    load_batch_data,
)
from advanced_furnace_ml.optimization import (
    bayesian_search,
    compare_optimizers,
    genetic_search,
    random_search,
)
from advanced_furnace_ml.recommendation import (
    build_recommendation_context,
    fit_feasibility_reference,
    recommend_or_fallback,
    safety_gate,
    score_candidates,
)


EXCEL = Path(__file__).parents[2] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class BowlModel:
    def __init__(self, offset=0.0):
        self.offset = offset

    def predict(self, frame):
        center = np.array([35.0, 0.65, 14.0, 70.0])
        scale = np.array([10.0, 1.0, 5.0, 50.0])
        values = frame[[SOLID_COL, WAIT_COL, DOOR_COUNT_COL, DOOR_DURATION_COL]].to_numpy()
        return 2500.0 + self.offset + 50.0 * (((values - center) / scale) ** 2).sum(axis=1)


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL)
        cls.context = build_recommendation_context(cls.df, 86000.0, trust_ratio=0.10)
        cls.feasibility = fit_feasibility_reference(cls.df)
        cls.model = BowlModel()
        cls.folds = [BowlModel(-20), BowlModel(0), BowlModel(20)]

    def test_optimizers_share_bounds_budget_and_integer_rules(self):
        functions = (random_search, genetic_search, bayesian_search)
        results = [
            fn(self.model, self.folds, self.context, self.feasibility, budget=120, seed=7)
            for fn in functions
        ]
        for result in results:
            self.assertEqual(len(result), 120)
            for column, (low, high) in self.context.bounds.items():
                self.assertTrue(result[column].between(low, high).all(), column)
            self.assertTrue(np.allclose(result[DOOR_COUNT_COL] % 1, 0))
        repeat = random_search(self.model, self.folds, self.context, self.feasibility, budget=120, seed=7)
        self.assertTrue(results[0].equals(repeat))
        comparison = compare_optimizers(
            self.model, self.folds, self.context, self.feasibility,
            common_budget=120, seeds=[1, 2], keep_results=False,
        )
        self.assertEqual(set(comparison["runs"]["optimizer"]), {"random_search", "genetic_algorithm", "bayesian_optimization"})
        self.assertTrue((comparison["runs"]["evaluations"] == 120).all())

    def test_safety_requires_savings_consensus_feasibility_and_no_boundary(self):
        candidates = random_search(self.model, self.folds, self.context, self.feasibility, budget=2000, seed=3)
        interior = candidates.loc[~candidates["boundary_hit"]].sort_values("penalized_objective").iloc[0]
        decision = safety_gate(interior, self.context)
        self.assertTrue(decision.passes, decision.reasons)
        bad = interior.copy()
        bad["conservative_predicted_gas"] = self.context.actual_baseline_gas + 1
        self.assertFalse(safety_gate(bad, self.context).passes)
        recommendation = recommend_or_fallback(candidates, self.context)
        self.assertEqual(recommendation["source"], "model_optimization")
        self.assertGreater(recommendation["estimated_saving_vs_actual_baseline"], 0)


if __name__ == "__main__":
    unittest.main()
