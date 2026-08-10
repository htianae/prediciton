from pathlib import Path
import pickle
import unittest

import numpy as np

from advanced_furnace_ml.data import FEATURE_COLS, TARGET_COL, load_batch_data
from advanced_furnace_ml.models import (
    ResidualBoostRegressor,
    RouteRegressor,
    WeightedOOFEnsemble,
    build_base_models,
    fit_nonnegative_ensemble_weights,
)


EXCEL = Path(__file__).parents[2] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL).iloc[:80]
        cls.X = cls.df[FEATURE_COLS]
        cls.y = cls.df[TARGET_COL]

    def test_required_models_and_routes(self):
        models = build_base_models(seed=7)
        expected = {"Ridge", "ElasticNet", "Huber", "GAM", "GPR", "CatBoost", "LightGBM", "Ridge+LGBMResidual", "Huber+LGBMResidual"}
        self.assertEqual(set(models), expected)
        for route in ("direct", "unit"):
            wrapped = RouteRegressor(models["Ridge"], route=route).fit(self.X, self.y)
            prediction = wrapped.predict(self.X.iloc[:3])
            self.assertEqual(prediction.shape, (3,))
            self.assertTrue(np.isfinite(prediction).all())
        gpr = RouteRegressor(models["GPR"], route="direct").fit(self.X, self.y)
        native_std = gpr.predict_std(self.X.iloc[:3])
        self.assertEqual(native_std.shape, (3,))
        self.assertTrue((native_std >= 0).all())
        ridge = RouteRegressor(models["Ridge"], route="direct").fit(self.X, self.y)
        with self.assertRaisesRegex(TypeError, "不支持"):
            ridge.predict_std(self.X.iloc[:3])

    def test_residual_and_weighted_ensembles_serialize(self):
        models = build_base_models(seed=7)
        residual = ResidualBoostRegressor(models["Ridge"], models["LightGBM"]).fit(self.X, self.y)
        pickle.loads(pickle.dumps(residual)).predict(self.X.iloc[:2])
        members = [RouteRegressor(models[name], "direct").fit(self.X, self.y) for name in ("Ridge", "Huber")]
        ensemble = WeightedOOFEnsemble(members, np.array([0.7, 0.3]))
        self.assertEqual(ensemble.predict(self.X.iloc[:2]).shape, (2,))
        weights = fit_nonnegative_ensemble_weights(
            np.column_stack([self.y.to_numpy(), self.y.to_numpy() + 10]), self.y.to_numpy()
        )
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
