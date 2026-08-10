from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from furnace_champion.artifact import (
    build_artifact,
    load_artifact,
    predict_with_interval,
    save_artifact,
    validate_input,
)
from furnace_champion.data import FEATURE_COLS, load_batch_data
from furnace_champion.modeling import evaluate_models_time_series, fit_champion


EXCEL = Path(__file__).parents[1] / "4_month_data_2026_02_01_2026_06_25.xlsx"


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_batch_data(EXCEL)
        cls.evaluation = evaluate_models_time_series(cls.df)
        cls.model_name = "Linear"
        cls.bundle = build_artifact(
            cls.df,
            cls.model_name,
            fit_champion(cls.df, cls.model_name),
            cls.evaluation["cv_models"][cls.model_name],
            cls.evaluation["oof_predictions"][cls.model_name],
            cls.evaluation["summary"],
        )

    def test_schema_validation_rejects_missing_extra_and_non_numeric(self):
        row = self.df[FEATURE_COLS].iloc[[0]].copy()
        self.assertEqual(list(validate_input(self.bundle, row).columns), FEATURE_COLS)
        with self.assertRaisesRegex(ValueError, "缺少"):
            validate_input(self.bundle, row.drop(columns=FEATURE_COLS[0]))
        with self.assertRaisesRegex(ValueError, "额外"):
            validate_input(self.bundle, row.assign(extra=1))
        bad = row.astype(object).copy()
        bad.iloc[0, 0] = "bad"
        with self.assertRaisesRegex(ValueError, "数值"):
            validate_input(self.bundle, bad)

    def test_prediction_has_interval_and_ood_warning(self):
        normal = predict_with_interval(self.bundle, self.df[FEATURE_COLS].iloc[[10]])
        self.assertGreaterEqual(normal.loc[0, "prediction_lower_90"], 0)
        self.assertGreater(normal.loc[0, "prediction_upper_90"], normal.loc[0, "prediction_lower_90"])
        extreme = self.df[FEATURE_COLS].iloc[[10]].copy()
        extreme[FEATURE_COLS[0]] = 999999
        result = predict_with_interval(self.bundle, extreme)
        self.assertTrue(bool(result.loc[0, "is_ood"]))
        self.assertIn(FEATURE_COLS[0], result.loc[0, "ood_features"])

    def test_joblib_round_trip_preserves_predictions(self):
        rows = self.df[FEATURE_COLS].iloc[:3]
        before = predict_with_interval(self.bundle, rows)["predicted_gas"].to_numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.joblib"
            save_artifact(self.bundle, path)
            loaded = load_artifact(path)
            after = predict_with_interval(loaded, rows)["predicted_gas"].to_numpy()
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
